# mypy: disable-error-code="type-arg, valid-type"
from dataclasses import dataclass
from typing import Optional
from gateforge.concepts import Bus, Interface
from gateforge.core import Expression, InputNet, Net, OutputNet, Reg, Wire
from gateforge.dsl import _else, _elseif, _if, always, always_comb, concat, cond, const, \
    namespace, reg, wire

from riscv.alu import Alu
from riscv.instruction_set import SynthesizeDecompressor


@dataclass
class RiscvParams:
    # Generate debug ports to expose internal state.
    debug: bool = False
    # RV32C profile
    isCompressedIsa = False
    # RV32E profile
    isEmbedded: bool = False
    hasEbreak: bool = False


class DebugBus(Bus["DebugBus"]):
    pc: OutputNet[Wire]
    normalizedInsn: OutputNet[Wire, 32]
    ebreak: OutputNet[Reg]

    def CreatePorts(self, addrSize):
        with namespace("DebugBus"):
            self.Construct(normalizedInsn=wire("normalizedInsn", 32).output.port,
                           pc=wire("pc", addrSize + 2).output.port,
                           ebreak=reg("ebreak").output.port)


class MemoryInterface(Interface["MemoryInterface"]):
    # CPU-originated address is valid, transfer initiated by asserting this signal.
    valid: OutputNet[Wire]
    # Instruction fetch when true, data fetch otherwise.
    insn: OutputNet[Wire]
    # Requested data is ready on the dataRead lines.
    ready: InputNet[Wire]

    # Word address.
    address: OutputNet[Wire]
    # CPU-originated data for write operation.
    dataWrite: OutputNet[Wire, 32]
    # Memory-originated data for read operation.
    dataRead: InputNet[Wire, 32]
    # Indicates which bytes in a word are to be written, read operation when zero.
    writeMask: OutputNet[Wire, 4]


    def __init__(self, addrSize):
        self.addrSize = addrSize
        with namespace("MemoryInterface"):
            self.ConstructDefault(address=wire(addrSize).output)


class ControlInterface(Interface["ControlInterface"]):
    reset: InputNet[Wire]
    clk: InputNet[Wire]
    trap: OutputNet[Wire]


    def __init__(self):
        with namespace("ControlInterface"):
            self.ConstructDefault()


class RegFileInterface(Interface["RegFileInterface"]):
    """Dual port interface for registers file.
    """
    # active posedge
    clk: InputNet[Wire]
    writeEn: InputNet[Wire]
    # Address zero should be wired to zero value.
    readAddrA: InputNet[Wire]
    readAddrB: InputNet[Wire]
    # Writing to address zero does nothing.
    writeAddr: InputNet[Wire]
    writeData: InputNet[Wire, 32]
    readDataA: OutputNet[Wire, 32]
    readDataB: OutputNet[Wire, 32]


def CreateDefaultRegFile(regIdxBits: int) -> RegFileInterface:
    """Just use array of words. Hopefully is inferred to block RAM primitives.

    :param regIdxBits: Number of bits in address.
    """

    with namespace("RegFile"):
        iface = RegFileInterface.CreateDefault(
            readAddrA=wire("readAddrA", regIdxBits).input,
            readAddrB=wire("readAddrB", regIdxBits).input,
            writeAddr=wire("writeAddr", regIdxBits).input)

        # Make address zero be out of bounds by inverting address bits. Out of range array access is
        # valid in Verilog and results to zero on reading and NOP on writing.
        regs = reg("regs", 32).array((1 << regIdxBits) - 1)

        with always(iface.clk.posedge):
            with _if(iface.writeEn):
                regs[~iface.writeAddr] <<= iface.writeData

        iface.readDataA <<= regs[~iface.readAddrA]
        iface.readDataB <<= regs[~iface.readAddrB]

    return iface


class AluOp:
    # Values are direct bits mappings from an opcode.
    ADD =  const("4'b0000")
    SUB =  const("4'b1000")
    SLL =  const("4'b0001")
    SLT =  const("4'b0010")
    SLTU = const("4'b0011")
    XOR =  const("4'b0100")
    SRL =  const("4'b0101")
    SRA =  const("4'b1101")
    OR =   const("4'b0110")
    AND =  const("4'b0111")


class InsnDecoder:
    # Load instruction (LB, LBU, LH, LHU, LW)
    isLoad: Wire
    # Store instruction (SB, SH, SW)
    isStore: Wire
    # Transfer size for load or store instruction.
    transferByte: Wire
    transferHalfWord: Wire
    transferWord: Wire
    # Extend sign bits when loading byte or half-word.
    isLoadSigned: Wire

    isLui: Wire
    isEbreak: Wire

    # Immediate value if any.
    immediate: Reg[32]

    rs1Idx: Wire
    rs2Idx: Wire
    rdIdx: Wire

    # ALU operation.
    isAluOp: Wire
    # ALU operation code if `isAluOp` is true. See `AluOp`.
    aluOp: Wire[4]

    # Valid only if `isAluOp` true
    isAluSlt: Wire
    isAluSltu: Wire
    #XXX rest

    # True if ALU operation performed on rs1 and rs2. If false the second operand is immediate value
    # in most cases.
    isAluRegToReg: Wire


    _input: Net[(31, 2)]
    _regIdxBits: int


    def __init__(self, *, input: Expression[(31, 2)], regIdxBits: int):
        self._regIdxBits = regIdxBits
        with namespace("InsnDecoder"):
            self._input = wire("_input", (31, 2))
            self._input <<= input

            self.isLoad = wire("isLoad")
            self.isStore = wire("isStore")
            self.transferByte = wire("transferByte")
            self.transferHalfWord = wire("transferHalfWord")
            self.transferWord = wire("transferWord")
            self.isLoadSigned = wire("isLoadSigned")
            self.immediate = reg("immediate", 32)
            self.isLui = wire("isLui")
            self.rs1Idx = wire("rs1Idx", self._regIdxBits)
            self.rs2Idx = wire("rs2Idx", self._regIdxBits)
            self.rdIdx = wire("rdIdx", self._regIdxBits)
            self.isAluOp = wire("isAluOp")
            self.aluOp = wire("aluOp", 4)
            self.isAluRegToReg = wire("isAluRegToReg")
            self.isAluSlt = wire("isAluSlt")
            self.isAluSltu = wire("isAluSltu")
            self.isEbreak = wire("isEbreak")


    def __call__(self):
        self.isLoad <<= self._input[6:2] == const("5'b00000")
        self.isStore <<= self._input[6:2] == const("5'b01000")
        self.isLoadSigned <<= ~self._input[14]
        self.transferByte <<= self._input[13:12] == const("2'b00")
        self.transferHalfWord <<= self._input[13:12] == const("2'b01")
        self.transferWord <<= self._input[13:12] == const("2'b10")

        self.rs1Idx <<= self._input[15 + self._regIdxBits - 1 : 15]
        self.rs2Idx <<= self._input[20 + self._regIdxBits - 1 : 20]
        self.rdIdx <<= self._input[7 + self._regIdxBits - 1 : 7]

        with always_comb():
            with _if ((self._input[6:2] == const("5'b11001")) |
                      (self._input[6:2] == const("5'b00000")) |
                      (self._input[6:2] == const("5'b00100"))):
                # I-type
                self.immediate <<= self._input[31].replicate(20) % self._input[31:20]
            with _elseif ((self._input[6] == 0) & (self._input[4:2] == const("3'b101"))):
                # U-type
                self.immediate <<= self._input[31:12] % const("12'b0")
            with _elseif (self._input[6:2] == const("5'b01000")):
                # S-type
                self.immediate <<= self._input[31].replicate(20) % self._input[31:25] % self._input[11:7]
            with _else():
                #XXX
                self.immediate <<= 0

        self.isLui <<= self._input[6:2] == const("5'b01101")

        self.isAluOp <<= (self._input[6] == const("1'b0")) & (self._input[4:2] == const("3'b100"))
        self.isAluRegToReg <<= self._input[5] & ~self.isLui
        # Ignore bit 30 (set to zero in the result) when immediate operation (bit 5 is zero), bit 30
        # is part of immediate value in such case. SRAI is exception, XXX revise
        self.aluOp <<= concat(
            (self._input[5] | (self._input[14:12] == const("3'b101"))) & self._input[30],
             self._input[14:12])

        self.isAluSlt <<= self.aluOp[2:0] == AluOp.SLT[2:0]
        self.isAluSltu <<= self.aluOp[2:0] == AluOp.SLTU[2:0]

        self.isEbreak <<= self._input == const("30'b000000000001000000000000011100")


class RiscvCpu:
    params: RiscvParams
    # Number of bits PC register is aligned to (those last bits are not stored)
    pcAlignBits: int
    # Number of bits to address registers (depends on RV32E extension).
    regIdxBits: int

    ctrlIface: ControlInterface
    memIface: MemoryInterface
    regFileIface: RegFileInterface
    dbg: Optional[DebugBus]

    memAddress: Reg
    memValid: Reg
    memWData: Reg
    memWriteMask: Reg

    pc: Reg
    # PC set for next instruction.
    isPcSet: Reg

    # Pre-fetched 16 bits high-order half-word of next instruction. Used only with compressed ISA.
    insnHi: Reg[16]
    # `insnHi` has valid content.
    insnHiValid: Reg

    insn: Reg[32]

    insnDecoder: InsnDecoder

    # True when `insn` contains current instruction and `insnDecoder` has decoded parameters
    # available.
    insnFetched: Reg

    # Main execution steps below
    # Fetching registers (actually operands `rs1` and `rs2`) either from register file, decoded
    # immediate or PC.
    stateRegFetch: Reg
    # Fetching data from memory. May be omitted for unrelated instructions.
    stateDataFetch: Reg
    # Store data either to memory or register file. PC may be written as well for branch
    # instructions.
    stateWriteBack: Reg
    # Trap is asserted
    stateTrap: Reg

    # Register fetch data latched on register file interface.
    regFetchLatched: Reg

    # Aliases for register file interface
    rs1: Wire[32]
    rs2: Wire[32]

    alu: Alu

    rd: Reg[32]
    writeRd: Reg


    def __init__(self, *, params: RiscvParams, ctrlIface: ControlInterface,
                 memIface: MemoryInterface, regFileIface: Optional[RegFileInterface] = None,
                 debugBus: Optional[DebugBus] = None):
        """_summary_

        :param params:
        :param memIface: _description_
        :param debugBus: External debug bus. If not provided and `params.debug` is True, default one
            is created with all nets exposed to module ports.
        """
        self.params = params

        self.pcAlignBits = 1 if params.isCompressedIsa else 2
        self.regIdxBits = 4 if self.params.isEmbedded else 5

        self.ctrlIface = ctrlIface
        self.memIface = memIface
        self.regFileIface = regFileIface if regFileIface is not None else \
            CreateDefaultRegFile(self.regIdxBits)

        if params.debug:
            if debugBus is None:
                self.dbg = DebugBus()
                self.dbg.CreatePorts(self.memIface.addrSize)
            else:
                self.dbg = debugBus

        with namespace("RiscvCpu"):
            self._Setup()


    def _Setup(self):
        self.memAddress = reg("memAddress", self.memIface.addrSize)
        self.memValid = reg("memValid")
        #XXX direct mux?
        self.memDataWrite = reg("memDataWrite", 32)
        self.memWriteMask = reg("memWriteMask", 4)

        self.pc = reg("pc", self.memIface.addrSize + 2 - self.pcAlignBits)
        self.isPcSet = reg("isPcSet")
        if self.dbg is not None:
            self.dbg.pc <<= self.pc % const(0, self.pcAlignBits)

        self.insnFetched = reg("insnFetched")

        #XXXX
        self.memIface.internal.Assign(valid=self.memValid,
                                      insn=~self.insnFetched,#XXX
                                      address=self.memAddress,
                                      dataWrite=self.memDataWrite,
                                      writeMask=self.memWriteMask)

        self.insn = reg("insn", 32)

        if self.params.isCompressedIsa:
            self.insnHi = reg("insnHi", 16)
            self.insnHiValid = reg("insnHiValid")

            decompressedInsn = wire("decompressedInsn", 30)
            with always_comb():
                SynthesizeDecompressor(self.insn[15:0], decompressedInsn)

            # Either decompressed or just fetched.
            normalizedInsn = cond(self.insn[1:0] == 0b11, self.insn[31:2], decompressedInsn)
            if self.dbg is not None:
                self.dbg.normalizedInsn <<= normalizedInsn % const(0b11, 2)

            self.insnDecoder = InsnDecoder(input=normalizedInsn, regIdxBits=self.regIdxBits)

        else:
            if self.dbg is not None:
                self.dbg.normalizedInsn <<= self.insn
            self.insnDecoder = InsnDecoder(input=self.insn[31:2], regIdxBits=self.regIdxBits)


        self.stateRegFetch = reg("stateRegFetch")
        self.stateDataFetch = reg("stateDataFetch")
        self.stateWriteBack = reg("stateWriteBack")
        self.stateTrap = reg("stateTrap")

        self.regFetchLatched = reg("regFetchLatched")

        self.ctrlIface.trap <<= self.stateTrap

        self.rs1 = self.regFileIface.external.readDataA
        self.rs2 = self.regFileIface.external.readDataB

        aluInA = wire("aluInA", 32)
        aluInB = wire("aluInB", 32)
        aluIsSub = wire("aluIsSub")

        #XXX branching
        aluIsSub <<= self.insnDecoder.isAluOp & (self.insnDecoder.isAluSlt | self.insnDecoder.isAluSltu)#XXX

        aluInA <<= self.rs1 #XXX PC
        aluInB <<= cond(self.insnDecoder.isAluRegToReg, self.rs2, self.insnDecoder.immediate)

        self.alu = Alu(inA=aluInA, inB=aluInB, isSub=aluIsSub)

        self.rd = reg("rd", 32)
        self.writeRd = reg("writeRd")


    def __call__(self):

        self.insnDecoder()
        self.alu()

        self.regFileIface.external.clk <<= self.ctrlIface.clk
        self.regFileIface.external.readAddrA <<= self.insnDecoder.rs1Idx
        self.regFileIface.external.readAddrB <<= self.insnDecoder.rs2Idx
        self.regFileIface.external.writeAddr <<= self.insnDecoder.rdIdx
        self.regFileIface.external.writeData <<= self.rd
        self.regFileIface.external.writeEn <<= self.writeRd

        with always(self.ctrlIface.clk.posedge):
            with _if(self.ctrlIface.reset):
                self._HandleReset()
            with _elseif(self.stateTrap):
                #XXX
                pass
            with _else():
                self._HandleState()


    def _HandleReset(self):

        self.insnFetched <<= False
        self.pc <<= 0
        if self.params.isCompressedIsa:
            self.insnHiValid <<= False

        self.stateRegFetch <<= False
        self.stateDataFetch <<= False
        self.stateWriteBack <<= False

        self.regFetchLatched <<= False

        self._FetchInstruction()


    def _HandleState(self):

        with _if(self.insnFetched):
            self._HandleInstruction()

        with _else():
            self._HandleInstructionFetch()


    def _HandleInstruction(self):

        # XXX if not branching
        with _if(~self.isPcSet):
            if self.params.isCompressedIsa:
                #XXX
                pass
            else:
                self.pc <<= self.pc + 1 #XXX
            self.isPcSet <<= True

        with _if (self.stateRegFetch):
            self._HandleRegFetch()

        with _elseif (self.stateDataFetch):
            #XXX
            pass

        with _elseif (self.stateWriteBack):
            self._HandleWriteBack()


    # Initiate fetched instruction execution. `insn` and decoder output will be available on next
    # clock.
    def _ExecuteInstruction(self):
        self.stateRegFetch <<= True
        self.isPcSet <<= False


    def _HandleRegFetch(self):

        if self.params.hasEbreak:
            with _if (self.insnDecoder.isEbreak):
                self.stateTrap <<= True
                if self.dbg is not None:
                    self.dbg.ebreak <<= True

            with _else ():
                self._FetchRegs()

        else:
            self._FetchRegs()


    def _FetchRegs(self):
        #XXX handle PC ops

        with _if (self.insnDecoder.isLui):
            self.stateWriteBack <<= True
            self.stateRegFetch <<= False


        with _elseif (self.regFetchLatched):

            self.stateRegFetch <<= False
            self.regFetchLatched <<= False
            self.stateTrap <<= True#XXX

        with _else ():
            self.regFetchLatched <<= True


    def _HandleWriteBack(self):
        with _if (self.insnDecoder.isLui):
            self.rd <<= self.alu.outAddSub

        # Register written by a single clock
        self.writeRd <<= True
        self.stateWriteBack <<= False

        self.stateTrap <<= True#XXX


    # Initiate instruction fetching
    def _FetchInstruction(self):
        if not self.params.isCompressedIsa:
            self.memAddress <<= self.pc
            self.memValid <<= True
            return


    def _HandleInstructionFetch(self):
        with _if(self.memIface.ready):

            self.memValid <<= False

            if not self.params.isCompressedIsa:
                self.insn <<= self.memIface.dataRead
                self.insnFetched <<= True
                self._ExecuteInstruction()
                return

            #XXX
