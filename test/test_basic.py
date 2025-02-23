import struct
from typing import Sequence
import unittest

from gateforge.compiler import CompileModule, CompileResult
from gateforge.core import RenderOptions
from riscv.instruction_set import Assemble as asm
from testbench import TestbenchModule
from utils import GetVerilatorParams, NullOutput, disableVerilatorTests, workspaceDir


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


    def WriteBytes(self, address: int, data: bytes):
        for i in range(len(data)):
            self.buf[address + i] = data[i]


    def ReadWord(self, address) -> int:
        return struct.unpack("<I", self.buf[address: address + 4])[0]


    def ReadByte(self, address) -> int:
        return int(self.buf[address])


    def HandleSim(self, ports):
        if ports.memValid != 1:
            ports.memDataRead = 0
            ports.memReady = False
            return
        mask = ports.memWriteMask
        if mask == 0:
            ports.memDataRead = self.Read(ports.memAddress)
        else:
            self.Write(ports.memAddress, ports.memDataWrite, mask)
        ports.memReady = True


class TestBase(unittest.TestCase):
    hasCompressedIsa = False
    result: CompileResult


    @classmethod
    def setUpClass(cls):
        cls.result = CompileModule(TestbenchModule, NullOutput(),
                                   renderOptions=RenderOptions(sourceMap=True),
                                   verilatorParams=GetVerilatorParams(),
                                   moduleKwargs={"hasCompressedIsa": cls.hasCompressedIsa})


    @classmethod
    def tearDownClass(cls):
        cls.result.simulationModel.Close()


    def setUp(self):
        self.sim = self.result.simulationModel
        self.sim.Reload()
        self.ports = self.sim.ports
        self.memSize = 16 * 1024 * 1024
        self.mem = Memory(self.memSize)
        self.clk = 0
        self.sim.Eval()
        self.sim.OpenVcd(workspaceDir / "test.vcd")


    def Tick(self):
        self.clk = 1 - self.clk
        self.ports.clk = self.clk
        self.mem.HandleSim(self.ports)
        self.sim.TimeInc(1)
        self.sim.Eval()
        self.sim.DumpVcd()


    def Reset(self):
        self.ports.reset = True
        self.Tick()
        self.Tick()
        self.ports.reset = False


    def SetProgram(self, program: Sequence[bytes], reset = True):
        # Convert to little-endian
        opCodes = list(map(lambda op: bytes(reversed(op)), program))
        offset = 0
        for opCode in opCodes:
            print(f"{offset:04x}: {opCode.hex(" ", 1)}")
            offset += len(opCode)
        data = b"".join(opCodes)
        self.programSize = len(data)
        self.mem.WriteBytes(0, data)

        if reset:
            self.Reset()


    def WaitEbreak(self):
        while self.ports.ebreak == 0:
            if self.ports.trap != 0:
                self.fail("Unexpected trap")
            self.Tick()


TEST_DATA_ADDR=0x102408


def LuiAddiImm(value: int) -> tuple[int, int]:
    """
    :return: Values to pass as immediate value to LUI/ADDI commands to get the specified
        32 bits immediate value in the target register at the end.
    """
    value = value & 0xFFFFFFFF
    luiImm = (value >> 12) & 0xFFFFF # Upper 20 bits
    addiImm = value & 0xFFF # Lower 12 bits

    # Check if lower 12 bits represent a negative value in 12-bit context

    if (addiImm & 0x800) and (value >> 31):
        # If sign bit set in 12-bit and original value negative
        luiImm = (luiImm + 1) & 0xFFFFF # Adjust upper immediate
        addiImm -= 0x1000 # Convert to negative offset

    return luiImm << 12, addiImm


def LuiImm(value: int) -> int:
    return LuiAddiImm(value)[0]

def AddiImm(value: int) -> int:
    return LuiAddiImm(value)[1]


@unittest.skipIf(disableVerilatorTests, "Verilator")
class Test(TestBase):

    def test_sw(self):

        testValue = 0xdeadbeef
        offset = 0x14

        self.SetProgram([
            asm("LUI", imm=LuiImm(TEST_DATA_ADDR), rd=10),
            asm("ADDI", imm=AddiImm(TEST_DATA_ADDR), rs1=10, rd=10),
            asm("LUI", imm=LuiImm(testValue), rd=11),
            asm("ADDI", imm=AddiImm(testValue), rs1=11, rd=11),
            asm("SW", imm=offset, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual(testValue, self.mem.ReadWord(TEST_DATA_ADDR + offset))


    def test_sh(self):

        testValue = 0xdeadbeef
        offset = 0x14

        self.SetProgram([
            asm("LUI", imm=LuiImm(TEST_DATA_ADDR), rd=10),
            asm("ADDI", imm=AddiImm(TEST_DATA_ADDR), rs1=10, rd=10),
            asm("LUI", imm=LuiImm(testValue), rd=11),
            asm("ADDI", imm=AddiImm(testValue), rs1=11, rd=11),
            asm("SH", imm=offset, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual(testValue & 0xffff, self.mem.ReadWord(TEST_DATA_ADDR + offset))


    def test_sh_hi(self):

        testValue = 0xdeadbeef
        offset = 0x16

        self.SetProgram([
            asm("LUI", imm=LuiImm(TEST_DATA_ADDR), rd=10),
            asm("ADDI", imm=AddiImm(TEST_DATA_ADDR), rs1=10, rd=10),
            asm("LUI", imm=LuiImm(testValue), rd=11),
            asm("ADDI", imm=AddiImm(testValue), rs1=11, rd=11),
            asm("SH", imm=offset, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual((testValue & 0xffff) << 16, self.mem.ReadWord(TEST_DATA_ADDR + offset - 2))


    def test_sb(self):

        testValue = 0xdeadbeef
        offset = 0x14

        self.SetProgram([
            asm("LUI", imm=LuiImm(TEST_DATA_ADDR), rd=10),
            asm("ADDI", imm=AddiImm(TEST_DATA_ADDR), rs1=10, rd=10),
            asm("LUI", imm=LuiImm(testValue), rd=11),
            asm("ADDI", imm=AddiImm(testValue), rs1=11, rd=11),
            asm("SB", imm=offset, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual(testValue & 0xff, self.mem.ReadWord(TEST_DATA_ADDR + offset))


    def test_sb_1(self):

        testValue = 0xdeadbeef
        offset = 0x15

        self.SetProgram([
            asm("LUI", imm=LuiImm(TEST_DATA_ADDR), rd=10),
            asm("ADDI", imm=AddiImm(TEST_DATA_ADDR), rs1=10, rd=10),
            asm("LUI", imm=LuiImm(testValue), rd=11),
            asm("ADDI", imm=AddiImm(testValue), rs1=11, rd=11),
            asm("SB", imm=offset, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual((testValue & 0xff) << 8, self.mem.ReadWord(TEST_DATA_ADDR + offset - 1))


    def test_sb_2(self):

        testValue = 0xdeadbeef
        offset = 0x16

        self.SetProgram([
            asm("LUI", imm=LuiImm(TEST_DATA_ADDR), rd=10),
            asm("ADDI", imm=AddiImm(TEST_DATA_ADDR), rs1=10, rd=10),
            asm("LUI", imm=LuiImm(testValue), rd=11),
            asm("ADDI", imm=AddiImm(testValue), rs1=11, rd=11),
            asm("SB", imm=offset, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual(testValue & 0xff, self.mem.ReadByte(TEST_DATA_ADDR + offset))


    def test_sb_3(self):

        testValue = 0xdeadbeef
        offset = 0x17

        self.SetProgram([
            asm("LUI", imm=LuiImm(TEST_DATA_ADDR), rd=10),
            asm("ADDI", imm=AddiImm(TEST_DATA_ADDR), rs1=10, rd=10),
            asm("LUI", imm=LuiImm(testValue), rd=11),
            asm("ADDI", imm=AddiImm(testValue), rs1=11, rd=11),
            asm("SB", imm=offset, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual(testValue & 0xff, self.mem.ReadByte(TEST_DATA_ADDR + offset))


@unittest.skipIf(disableVerilatorTests, "Verilator")
class TestUncompressedOnCompressedIsa(Test):
    hasCompressedIsa = True


@unittest.skipIf(disableVerilatorTests, "Verilator")
class TestCompressed(TestBase):
    hasCompressedIsa = True

    def test_c_li(self):

        testValue = 27
        offset = 0x14

        self.SetProgram([
            asm("LUI", imm=LuiImm(TEST_DATA_ADDR), rd=10),
            asm("ADDI", imm=AddiImm(TEST_DATA_ADDR), rs1=10, rd=10),
            asm("C.LI", imm=testValue, rd=11),
            asm("SW", imm=offset, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual(testValue, self.mem.ReadWord(TEST_DATA_ADDR + offset))


    def test_c_li_addi(self):

        testValue = 27
        offset = 0x14

        self.SetProgram([
            asm("LUI", imm=LuiImm(TEST_DATA_ADDR), rd=10),
            asm("ADDI", imm=AddiImm(TEST_DATA_ADDR), rs1=10, rd=10),
            asm("C.LI", imm=testValue, rd=11),
            asm("C.ADDI", imm=2, rd=11),
            asm("SW", imm=offset, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual(testValue + 2, self.mem.ReadWord(TEST_DATA_ADDR + offset))
