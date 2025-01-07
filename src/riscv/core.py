from dataclasses import dataclass
from typing import Optional
from gateforge.concepts import Bus, Interface
from gateforge.core import InputNet, OutputNet, Reg, Wire
from gateforge.dsl import _else, _if, always, const, namespace, reg, wire


@dataclass
class RiscvParams:
    # Generate debug ports to expose internal state.
    debug: bool = False


class DebugBus(Bus["DebugBus"]):
    pc: OutputNet[Wire]
    decodedInsn: OutputNet[Wire, 32]

    def CreatePorts(self, addrSize):
        with namespace("DebugBus"):
            self.Construct(decodedInsn=wire(32, "decodedInsn").output.port,
                           pc=wire(addrSize, "pc").output.port)


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

            self.memAddress = reg(memIface.addrSize, "memAddress")
            self.memValid = reg("memValid")
            #XXX direct mux?
            self.memDataWrite = reg(32, "memDataWrite")
            self.memWriteMask = reg(4, "memWriteMask")

            self.insnFetched = reg("insnFetched")

            self.memIface.internal.Assign(valid=self.memValid,
                                          insn=~self.insnFetched,
                                          address=self.memAddress,
                                          dataWrite=self.memDataWrite,
                                          writeMask=self.memWriteMask)

            self.pc = reg(self.memIface.addrSize, "pc")
            if self.dbg is not None:
                self.dbg.pc <<= self.pc

            #XXX
            self.insn = reg(32, "insn")
            if self.dbg is not None:
                self.dbg.decodedInsn <<=  self.insn


    def __call__(self):

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
