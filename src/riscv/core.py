# mypy: disable-error-code="type-arg, valid-type"
from dataclasses import dataclass
from typing import Optional
from gateforge.concepts import Bus, Interface
from gateforge.core import Expression, InputNet, Net, OutputNet, Reg, Wire
from gateforge.dsl import _else, _elseif, _if, always, always_comb, concat, const, namespace, reg, wire


@dataclass
class RiscvParams:
    # Generate debug ports to expose internal state.
    debug: bool = False
    # RV32C profile
    isCompressedIsa = False
    # RV32E profile
    isEmbedded: bool = False


class DebugBus(Bus["DebugBus"]):
    pc: OutputNet[Wire]
    decodedInsn: OutputNet[Wire, 32]

    def CreatePorts(self, addrSize):
        with namespace("DebugBus"):
            self.Construct(decodedInsn=wire("decodedInsn", 32).output.port,
                           pc=wire("pc", addrSize).output.port)


class MemoryInterface(Interface["MemoryInterface"]):
    # CPU-originated address and data valid
    valid: OutputNet[Wire]
    # Instruction fetch when true, data fetch otherwise
    insn: OutputNet[Wire]
    # Requested data is ready on the rData lines
    ready: InputNet[Wire]

    address: OutputNet[Wire]
    dataWrite: OutputNet[Wire, 32]
    dataRead: InputNet[Wire, 32]
    writeMask: OutputNet[Wire, 4]


    def __init__(self, addrSize):
        self.addrSize = addrSize
        with namespace("MemoryInterface"):
            self.ConstructDefault(address=wire(addrSize).output)


class ControlInterface(Interface["ControlInterface"]):
    reset: InputNet[Wire]
    clk: InputNet[Wire]


    def __init__(self):
        with namespace("ControlInterface"):
            self.ConstructDefault()


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
    # Immediate value if any.
    immediate: Reg[32]
    isLui: Wire
    rs1Idx: Wire
    rs2Idx: Wire
    rdIdx: Wire
    # ALU operation.
    isAluOp: Wire
    # ALU operation code if `isAluOp` is true.
    aluOp: Wire[4]
    # True operation with rs1 and immediate value, false if operation on rs1 and rs2.
    isAluImmediate: Wire
    isSlt: Wire
    isSltu: Wire

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
            self.isAluImmediate = wire("isAluImmediate")
            self.isSlt = wire("isSlt")
            self.isSltu = wire("isSltu")


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
        self.isAluImmediate <<= ~self._input[5]
        # Ignore bit 30 (set to zero in the result) when immediate operation (bit 5 is zero), bit 30
        # is part of immediate value in such case. SRAI is exception, XXX
        self.aluOp <<= concat(
            (self._input[5] | (self._input[14:12] == const("3'b101"))) & self._input[30],
             self._input[14:12])
        self.isSlt <<= self.aluOp[2:0] == AluOp.SLT[2:0]
        self.isSltu <<= self.aluOp[2:0] == AluOp.SLTU[2:0]


class RiscvCpu:
    params: RiscvParams
    ctrlIface: ControlInterface
    memIface: MemoryInterface
    dbg: Optional[DebugBus]

    pc: Reg
    insn: Reg
    insnFetched: Reg
    memAddress: Reg
    memValid: Reg
    memWData: Reg
    memWriteMask: Reg
    insnDecoder: InsnDecoder


    def __init__(self, *, params: RiscvParams, ctrlIface: ControlInterface,
                 memIface: MemoryInterface,
                 debugBus: Optional[DebugBus] = None):
        """_summary_

        :param params:
        :param memIface: _description_
        :param debugBus: External debug bus. If not provided and `params.debug` is True, default one
            is created with all nets exposed to module ports.
        """
        self.params = params
        self.ctrlIface = ctrlIface
        self.memIface = memIface

        if params.debug:
            if debugBus is None:
                self.dbg = DebugBus()
                self.dbg.CreatePorts(self.memIface.addrSize)
            else:
                self.dbg = debugBus

        with namespace("RiscvCpu"):

            self.memAddress = reg("memAddress", memIface.addrSize)
            self.memValid = reg("memValid")
            #XXX direct mux?
            self.memDataWrite = reg("memDataWrite", 32)
            self.memWriteMask = reg("memWriteMask", 4)

            self.insnFetched = reg("insnFetched")

            self.memIface.internal.Assign(valid=self.memValid,
                                          insn=~self.insnFetched,
                                          address=self.memAddress,
                                          dataWrite=self.memDataWrite,
                                          writeMask=self.memWriteMask)

            self.pc = reg("pc", self.memIface.addrSize)
            if self.dbg is not None:
                self.dbg.pc <<= self.pc

            #XXX
            self.insn = reg("insn", 32)
            if self.dbg is not None:
                self.dbg.decodedInsn <<= self.insn

            self.insnDecoder = InsnDecoder(input=self.insn[31:2],
                                           regIdxBits = 4 if self.params.isEmbedded else 5)


    def __call__(self):

        self.insnDecoder()

        with always(self.ctrlIface.clk.negedge):
            with _if(self.ctrlIface.reset):
                self._HandleReset()
            with _else():
                self._HandleState()


    def _HandleReset(self):
        self.insnFetched <<= False
        self.pc <<= 0


    def _HandleState(self):

        with _if(~self.insnFetched):
            self.memAddress <<= self.pc
            self.memValid <<= True

            with _if(self.memIface.ready):
                self.insn <<= self.memIface.dataRead
                self.insnFetched <<= True
                self.pc <<= self.pc + 1 #XXX
