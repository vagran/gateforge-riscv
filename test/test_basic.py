import io
from pathlib import Path
import struct
import unittest

from gateforge.compiler import CompileModule
from gateforge.core import RenderOptions
from gateforge.verilator import VerilatorParams
from testbench import TestbenchModule, disableVerilatorTests


class NullOutput(io.StringIO):
    def write(self, _, /):
        pass


class Memory:

    def __init__(self, size):
        self.size = size
        self.buf = bytearray(size)


    def _CheckAddress(self, address):
        if address * 4 >= self.size:
            raise Exception(f"Address out of range: {address:x}")


    def Read(self, address: int) -> int:
        self._CheckAddress(address)
        offset = address * 4
        return struct.unpack("<I", self.buf[offset:offset+4])[0]


    def Write(self, address: int, value: int, mask: int = 0xf):
        self._CheckAddress(address)
        offset = address * 4
        if mask != 0xf:
            mask = ((0xff000000 if (mask & 8) != 0 else 0) |
                    (0x00ff0000 if (mask & 4) != 0 else 0) |
                    (0x0000ff00 if (mask & 2) != 0 else 0) |
                    (0x000000ff if (mask & 1) != 0 else 0))
            value = (self.Read(address) & ~mask) | (value & mask)
        self.buf[offset:offset+4] = struct.pack("<I", value)


    def HandleSim(self, ports):
        if ports.memValid != 1:
            return
        mask = ports.memWriteMask
        if mask == 0:
            ports.memDataRead = self.Read(ports.memAddress)
        else:
            self.Write(ports.memAddress, ports.memDataWrite, mask)
        ports.memReady = 1


class TestBase(unittest.TestCase):
    def setUp(self):
        wspDir = Path(__file__).parent / "workspace"
        verilatorParams = VerilatorParams(buildDir=str(wspDir), quite=False)
        self.result = CompileModule(TestbenchModule, NullOutput(),
                                    renderOptions=RenderOptions(sourceMap=True),
                                    verilatorParams=verilatorParams)
        self.sim = self.result.simulationModel
        self.ports = self.sim.ports
        self.mem = Memory(0x10000)
        self.clk = 0
        self.sim.Eval()
        self.sim.OpenVcd(wspDir / "test.vcd")


    def tearDown(self):
        self.sim.Close()


    def Tick(self):
        self.clk = 1 - self.clk
        self.ports.clk = self.clk
        self.mem.HandleSim(self.ports)
        self.sim.TimeInc(1)
        self.sim.Eval()
        self.sim.DumpVcd()


@unittest.skipIf(disableVerilatorTests, "Verilator")
class Basic(TestBase):

    def test_basic(self):
        self.mem.Write(0, 0xdeadbeef)
        while self.ports.memInsn:
            self.Tick()
