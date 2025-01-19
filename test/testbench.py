from gateforge.dsl import wire
from riscv.core import RiscvParams, RiscvCpu, MemoryInterface, ControlInterface


class Testbench:
    cpu: RiscvCpu
    ctrlIface: ControlInterface
    memIface: MemoryInterface

    def __init__(self):
        params = RiscvParams(debug=True)
        self.ctrlIface = ControlInterface()
        self.memIface = MemoryInterface(16)
        self.cpu = RiscvCpu(params=params, ctrlIface=self.ctrlIface, memIface=self.memIface)


    def __call__(self):

        self.ctrlIface.external.Assign(
            reset=wire("reset").input.port,
            clk=wire("clk").input.port
        )

        self.memIface.external.Assign(
            valid=wire("memValid").output.port,
            insn=wire("memInsn").output.port,
            ready=wire("memReady").input.port,
            address=wire("memAddress", self.memIface.addrSize).output.port,
            dataWrite=wire("memDataWrite", 32).output.port,
            dataRead=wire("memDataRead", 32).input.port,
            writeMask=wire("memWriteMask", 4).output.port)

        self.cpu()

        return self


def TestbenchModule():
    return Testbench()()
