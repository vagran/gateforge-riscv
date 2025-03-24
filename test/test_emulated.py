from pathlib import Path
import struct
from typing import Sequence
import unittest

from gateforge.compiler import CompileModule, CompileResult
from gateforge.core import RenderOptions
from riscv.instruction_set import Assemble as asm
from testbench import TestbenchModule
from utils import GetVerilatorParams, NullOutput, SignExtend, disableVerilatorTests, workspaceDir


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


    def ReadBytes(self, address, size) -> bytes:
        return self.buf[address:address + size]


    def WriteWord(self, address, value):
        self.buf[address: address + 4] = struct.pack("<I", value)


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
                                   verilatorParams=GetVerilatorParams(cls.__name__),
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


    def SetProgram(self, program: Sequence[tuple[str, bytes]], reset = True):
        # Convert to little-endian

        print("======= Test code =======")
        offset = 0
        for asm, opCode in program:
            print(f"{offset:04x}: {bytes(reversed(opCode)).hex(" ", 1):12} {asm}")
            offset += len(opCode)

        data = b"".join(map(lambda op: bytes(reversed(op[1])), program))
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


def Li(value: int, rd: int) -> Sequence[tuple[str, bytes]]:
    return [
        asm("LUI", imm=LuiImm(value), rd=rd),
        asm("ADDI", imm=AddiImm(value), rs1=rd, rd=rd)
    ]


@unittest.skipIf(disableVerilatorTests, "Verilator")
class TestNoCompression(TestBase):

    def test_sw(self):

        testValue = 0xdeadbeef
        offset = 0x14

        self.SetProgram([
            *Li(TEST_DATA_ADDR, 10),
            *Li(testValue, 11),
            asm("SW", imm=offset, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual(testValue, self.mem.ReadWord(TEST_DATA_ADDR + offset))


    def test_sh(self):

        testValue = 0xdeadbeef
        offset = 0x14

        self.SetProgram([
            *Li(TEST_DATA_ADDR, 10),
            *Li(testValue, 11),
            asm("SH", imm=offset, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual(testValue & 0xffff, self.mem.ReadWord(TEST_DATA_ADDR + offset))


    def test_sh_hi(self):

        testValue = 0xdeadbeef
        offset = 0x16

        self.SetProgram([
            *Li(TEST_DATA_ADDR, 10),
            *Li(testValue, 11),
            asm("SH", imm=offset, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual((testValue & 0xffff) << 16, self.mem.ReadWord(TEST_DATA_ADDR + offset - 2))


    def test_sb(self):

        testValue = 0xdeadbeef
        offset = 0x14

        self.SetProgram([
            *Li(TEST_DATA_ADDR, 10),
            *Li(testValue, 11),
            asm("SB", imm=offset, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual(testValue & 0xff, self.mem.ReadWord(TEST_DATA_ADDR + offset))


    def test_sb_1(self):

        testValue = 0xdeadbeef
        offset = 0x14 + 1

        self.SetProgram([
            *Li(TEST_DATA_ADDR, 10),
            *Li(testValue, 11),
            asm("SB", imm=offset, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual((testValue & 0xff) << 8, self.mem.ReadWord(TEST_DATA_ADDR + offset - 1))


    def test_sb_2(self):

        testValue = 0xdeadbeef
        offset = 0x14 + 2

        self.SetProgram([
            *Li(TEST_DATA_ADDR, 10),
            *Li(testValue, 11),
            asm("SB", imm=offset, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual(testValue & 0xff, self.mem.ReadByte(TEST_DATA_ADDR + offset))


    def test_sb_3(self):

        testValue = 0xdeadbeef
        offset = 0x14 + 3

        self.SetProgram([
            *Li(TEST_DATA_ADDR, 10),
            *Li(testValue, 11),
            asm("SB", imm=offset, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual(testValue & 0xff, self.mem.ReadByte(TEST_DATA_ADDR + offset))


    def test_alu(self):
        v1 = 0xdeadbeef
        v2 = 0xc001babe
        v3 = 0xa55aa55a
        v4 = 42
        v5 = 37
        v6 = 0x7bc

        self.SetProgram([
            *Li(TEST_DATA_ADDR, 10),
            *Li(v1, 11),
            *Li(v2, 12),
            *Li(v3, 13),
            asm("ADD", rs1=11, rs2=12, rd=11),
            asm("XOR", rs1=11, rs2=13, rd=11),
            asm("AND", rs1=12, rs2=13, rd=12),
            asm("OR", rs1=11, rs2=12, rd=11),
            asm("ADDI", imm=v4, rs1=11, rd=11),
            asm("XORI", imm=v5, rs1=11, rd=11),
            asm("ORI", imm=v6, rs1=11, rd=11),
            asm("XORI", imm=0xfff, rs1=11, rd=11), # Sign-extends immediate, so it inverts all bits
            asm("SW", imm=0, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        r11 = v1
        r12 = v2
        r13 = v3
        r11 = (r11 + r12) & 0xffffffff
        r11 = r11 ^ r13
        r12 = r12 & r13
        r11 = r11 | r12
        r11 = (r11 + v4) & 0xffffffff
        r11 = r11 ^ v5
        r11 = r11 | v6
        r11 = r11 ^ 0xffffffff

        self.WaitEbreak()

        self.assertEqual(r11, self.mem.ReadWord(TEST_DATA_ADDR))


    def _TestSlt(self, v1, v2, unsigned=False):
        self.SetProgram([
            *Li(TEST_DATA_ADDR, 10),
            *Li(v1, 11),
            *Li(v2, 12),
            asm("SLTU" if unsigned else "SLT", rs1=11, rs2=12, rd=13),
            asm("SW", imm=0, rs1=10, rs2=13),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual(1 if v1 < v2 else 0, self.mem.ReadWord(TEST_DATA_ADDR))


    def _TestSlti(self, v1, v2, unsigned=False):
        self.SetProgram([
            *Li(TEST_DATA_ADDR, 10),
            *Li(v1, 11),
            asm("SLTIU" if unsigned else "SLTI", rs1=11, imm=v2, rd=13),
            asm("SW", imm=0, rs1=10, rs2=13),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual(1 if v1 < v2 else 0, self.mem.ReadWord(TEST_DATA_ADDR))


    def test_slt_positive_less(self):
        self._TestSlt(10, 20)


    def test_slt_positive_equal(self):
        self._TestSlt(10, 10)


    def test_slt_positive_greater(self):
        self._TestSlt(20, 10)


    def test_slt_neg_pos_less(self):
        self._TestSlt(-20, 10)


    def test_slt_neg_pos_greater(self):
        self._TestSlt(10, -20)


    def test_slt_negative_equal(self):
        self._TestSlt(-10, -10)


    def test_slt_negative_less(self):
        self._TestSlt(-20, -10)


    def test_slt_negative_greater(self):
        self._TestSlt(-10, -20)


    def test_sltu_positive_less(self):
        self._TestSlt(10, 20, True)


    def test_sltu_positive_equal(self):
        self._TestSlt(10, 10, True)


    def test_sltu_positive_greater(self):
        self._TestSlt(20, 10, True)


    def test_sltu_neg_pos_less(self):
        self._TestSlt(10, 0xf0000000, True)


    def test_sltu_neg_pos_greater(self):
        self._TestSlt(0xf0000000, 10, True)


    def test_sltu_negative_less(self):
        self._TestSlt(0xf0000000, 0xf1000000, True)


    def test_sltu_negative_greater(self):
        self._TestSlt(0xf1000000, 0xf0000000, True)


    def test_sltu_negative_equal(self):
        self._TestSlt(0xf0000000, 0xf0000000, True)


    def test_slti_positive_less(self):
        self._TestSlti(10, 20)


    def test_slti_positive_equal(self):
        self._TestSlti(10, 10)


    def test_slti_positive_greater(self):
        self._TestSlti(20, 10)


    def test_slti_neg_pos_less(self):
        self._TestSlti(-20, 10)


    def test_slti_neg_pos_greater(self):
        self._TestSlti(10, -20)


    def test_slti_negative_equal(self):
        self._TestSlti(-10, -10)


    def test_slti_negative_less(self):
        self._TestSlti(-20, -10)


    def test_slti_negative_greater(self):
        self._TestSlti(-10, -20)


    def test_sltiu_positive_less(self):
        self._TestSlti(10, 20, True)


    def test_sltiu_positive_equal(self):
        self._TestSlti(10, 10, True)


    def test_sltiu_positive_greater(self):
        self._TestSlti(20, 10, True)


    def test_sltiu_neg_pos_less(self):
        self._TestSlti(10, 0xf00, True)


    def test_sltiu_neg_pos_greater(self):
        self._TestSlti(0xf0000000, 10, True)


    def test_sltiu_negative_less(self):
        self._TestSlti(0xf0000000, 0xffffff10, True)


    def test_sltiu_negative_greater(self):
        self._TestSlti(0xffffff10, 0xffffff00, True)


    def test_sltiu_negative_equal(self):
        self._TestSlti(0xffffff00, 0xffffff00, True)


    def _TestShift(self, v1, v2, op, imm=False):
        program = [
            *Li(TEST_DATA_ADDR, 10),
            *Li(v1, 11)
        ]
        if imm:
            program.append(asm(op + "I", rs1=11, imm=v2, rd=13))
        else:
            program.extend([
                *Li(v2, 12),
                asm(op, rs1=11, rs2=12, rd=13)
            ])
        program.extend([
            asm("SW", imm=0, rs1=10, rs2=13),
            asm("EBREAK")
        ])
        self.SetProgram(program)
        self.WaitEbreak()

        if op == "SLL":
            result = (v1 << v2) & 0xffffffff
        elif op == "SRL":
            result = (v1 >> v2) & ((1 << (32 - v2)) - 1)
        else:
            result = (SignExtend(v1) >> v2) & 0xffffffff

        self.assertEqual(result, self.mem.ReadWord(TEST_DATA_ADDR))


    def test_sll(self):
        self._TestShift(0xdeadbeef, 5, "SLL")


    def test_srl(self):
        self._TestShift(0xdeadbeef, 5, "SRL")


    def test_sra(self):
        self._TestShift(0xdeadbeef, 5, "SRA")


    def test_slli(self):
        self._TestShift(0xdeadbeef, 5, "SLL", True)


    def test_srli(self):
        self._TestShift(0xdeadbeef, 5, "SRL", True)


    def test_srai(self):
        self._TestShift(0xdeadbeef, 5, "SRA", True)


    def _TestLoad(self, value: int, size: int, offset: int, isSigned: bool):
        if size == 8:
            op = "LB" if isSigned else "LBU"
        elif size == 16:
            op = "LH" if isSigned else "LHU"
        elif size == 32:
            op = "LW"
        else:
            raise Exception("Bad size")
        mask = (1 << size) - 1
        value = value & mask
        baseOffset = 0x14
        self.mem.WriteWord(TEST_DATA_ADDR + baseOffset, value << (offset * 8))

        self.SetProgram([
            *Li(TEST_DATA_ADDR, 10),
            asm(op, rs1=10, imm=baseOffset + offset, rd=11),
            asm("SW", imm=0, rs1=10, rs2=11),
            asm("EBREAK")
        ])
        self.WaitEbreak()

        if isSigned:
            signBit = 1 << (size - 1)
            if value & signBit != 0:
                value = value | (0xffffffff & ~mask)
        self.assertEqual(value, self.mem.ReadWord(TEST_DATA_ADDR))


    def test_lw(self):
        self._TestLoad(0xdeadbeef, 32, 0, False)


    def test_lhu(self):
        self._TestLoad(0xabcd, 16, 0, False)


    def test_lhu_1(self):
        self._TestLoad(0xabcd, 16, 2, False)


    def test_lh(self):
        self._TestLoad(0xabcd, 16, 0, True)


    def test_lh_1(self):
        self._TestLoad(0xabcd, 16, 2, True)


    def test_lbu(self):
        self._TestLoad(0xab, 8, 0, False)


    def test_lbu_1(self):
        self._TestLoad(0xab, 8, 1, False)


    def test_lbu_2(self):
        self._TestLoad(0xab, 8, 2, False)


    def test_lbu_3(self):
        self._TestLoad(0xab, 8, 3, False)


    def test_lb(self):
        self._TestLoad(0xab, 8, 0, True)


    def test_lb_1(self):
        self._TestLoad(0xab, 8, 1, True)


    def test_lb_2(self):
        self._TestLoad(0xab, 8, 2, True)


    def test_lb_3(self):
        self._TestLoad(0xab, 8, 3, True)


    def test_auipc(self):
        offset = 42 * 0x1000
        self.SetProgram([
            *Li(TEST_DATA_ADDR, 10),
            asm("AUIPC", imm=offset, rd=11),
            asm("SW", imm=0, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual(offset + 2 * 4, self.mem.ReadWord(TEST_DATA_ADDR))


    def test_jal(self):
        testValue = 0xdeadbeef
        self.SetProgram([
            *Li(TEST_DATA_ADDR, 10), # 0
            *Li(testValue, 11), # 2
            asm("JAL", imm=(6 - 4) * 4, rd=12), # 4
            asm("XOR", rs1=11, rs2=11, rd=11), # 5 - return address
            asm("SW", imm=0, rs1=10, rs2=11), # 6 - jump here
            asm("SW", imm=4, rs1=10, rs2=12),
            asm("EBREAK")
        ])

        self.WaitEbreak()
        self.assertEqual(testValue, self.mem.ReadWord(TEST_DATA_ADDR))
        # Return address
        self.assertEqual(5 * 4, self.mem.ReadWord(TEST_DATA_ADDR + 4))


    def test_jalr(self):
        testValue = 0xdeadbeef
        self.SetProgram([
            *Li(TEST_DATA_ADDR, 10), # 0
            *Li(testValue, 11), # 2
            *Li(6 * 4, 13), # 4
            asm("JALR", imm=(8 - 6) * 4, rs1=13, rd=12), # 6
            asm("XOR", rs1=11, rs2=11, rd=11), # 7 - return address
            asm("SW", imm=0, rs1=10, rs2=11), # 8 - jump here
            asm("SW", imm=4, rs1=10, rs2=12),
            asm("EBREAK")
        ])

        self.WaitEbreak()
        self.assertEqual(testValue, self.mem.ReadWord(TEST_DATA_ADDR))
        # Return address
        self.assertEqual(7 * 4, self.mem.ReadWord(TEST_DATA_ADDR + 4))


    def _TestBranchingImpl(self, value1: int, value2: int, opName: str, op):
        testValue = 0xdeadbeef
        self.SetProgram([
            *Li(TEST_DATA_ADDR, 10),
            *Li(testValue, 11),
            *Li(value1, 12),
            *Li(value2, 13),
            asm("B" + opName, imm=8, rs1=12, rs2=13),
            asm("XORI", imm=-1, rs1=11, rd=11), # Zeroes mark
            asm("SW", imm=0, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        mark = self.mem.ReadWord(TEST_DATA_ADDR)
        try:
            if op(value1, value2):
                self.assertEqual(testValue, mark)
            else:
                self.assertEqual(~testValue & 0xffffffff, mark)
        except:
            print(f"Failed op: {value1} {opName} {value2}")
            raise


    def _TestBranching(self, value1: int, value2: int):
        ops = [
            ("EQ", lambda x, y: (x & 0xffffffff) == (y & 0xffffffff)),
            ("NE", lambda x, y: (x & 0xffffffff) != (y & 0xffffffff)),
            ("LT", lambda x, y: SignExtend(x) < SignExtend(y)),
            ("GE", lambda x, y: SignExtend(x) >= SignExtend(y)),
            ("LTU", lambda x, y: (x & 0xffffffff) < (y & 0xffffffff)),
            ("GEU", lambda x, y: (x & 0xffffffff) >= (y & 0xffffffff))
        ]

        for opName, op in ops:
            self._TestBranchingImpl(value1, value2, opName, op)


    def test_branching(self):
        values = [
            (0, 0),
            (1, 1),
            (100, 0),
            (100, 50),
            (0x80000000, 0),
            (0x80000000, 100),
            (0x80000000, 0x80000000),
            (0xffffffff, 0),
            (0xffffffff, 100),
            (0xffffffff, 0x80000000),
            (0xffffffff, 0xffffffff)
        ]

        for x, y in values:
            self._TestBranching(x, y)
            if (x != y):
                self._TestBranching(y, x)
            if x != SignExtend(x) & 0xffffffff:
                self._TestBranching(-x, y)
                if (x != y):
                    self._TestBranching(y, -x)
            if y != SignExtend(y) & 0xffffffff:
                self._TestBranching(x, -y)
                if (x != y):
                    self._TestBranching(-y, x)
            if x != SignExtend(x) & 0xffffffff and y != SignExtend(y) & 0xffffffff:
                self._TestBranching(-x, -y)
                if (x != y):
                    self._TestBranching(-y, -x)


    def test_auipc_after_branching(self):
        testValue = 0xdeadbeef
        self.SetProgram([
            *Li(TEST_DATA_ADDR, 10), # 0
            *Li(testValue, 11), # 2
            asm("XOR", rs1=12, rs2=12, rd=12), # 4
            asm("BEQ", imm=4 * 4, rs1=12, rs2=0), # 5
            asm("XOR", rs1=11, rs2=11, rd=11), # 6
            asm("XOR", rs1=11, rs2=11, rd=11), # 7
            asm("XOR", rs1=11, rs2=11, rd=11), # 8
            asm("AUIPC", imm=0x2000, rd=13), # 9
            asm("ADD", rs1=11, rs2=13, rd=11),
            asm("SW", imm=0, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()
        self.assertEqual(testValue + 9 * 4 + 0x2000, self.mem.ReadWord(TEST_DATA_ADDR))


    def TestSoftMul(self, x, y):

        def mul(a0, a1):
            a0 = a0 & 0xffff_ffff
            a1 = a1 & 0xffff_ffff
            a2 = 0
            if a0 == 0:
                return a2
            while True:
                a3 = (a0 << 31) & 0xffff_ffff
                # Arithmetic shift right by 31 bits, effectively sets all bits to sign bit
                a3 = 0 if a3 == 0 else 0xffff_ffff
                a3 = a3 & a1
                a2 = (a2 + a3) & 0xffff_ffff
                a0 = a0 >> 1
                a1 = a1 << 1
                if a0 == 0:
                    break
            return a2

        a0 = 10
        a1 = 11
        a2 = 12
        a3 = 13

        self.SetProgram([
            *Li(x, a0), # 0
            *Li(y, a1), # 2
            *Li(0, a2), # 4
            asm("BEQ", rs1=0, rs2=a0, imm=(13-5)*4), # 5
            asm("SLLI", rd=a3, rs1=a0, imm=31), # 6
            asm("SRAI", rd=a3, rs1=a3, imm=31), # 7
            asm("AND", rd=a3, rs1=a3, rs2=a1), # 8
            asm("ADD", rd=a2, rs1=a3, rs2=a2), # 9
            asm("SRLI", rd=a0, rs1=a0, imm=1), # 10
            asm("SLLI", rd=a1, rs1=a1, imm=1), # 11
            asm("BNE", rs1=0, rs2=a0, imm=(6-12)*4), # 12
            *Li(TEST_DATA_ADDR, 14), # 13
            asm("SW", imm=0, rs1=14, rs2=a2),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        expected = (x * y) & 0xFFFFFFFF
        self.assertEqual(expected, mul(x, y))
        self.assertEqual(expected, self.mem.ReadWord(TEST_DATA_ADDR))


    def test_soft_mul(self):
        self.TestSoftMul(0xdeadbeef, 0xcc9e2d51)

    def test_soft_mul_0(self):
        self.TestSoftMul(0xdeadbeef, 0)

    def test_soft_mul_0_1(self):
        self.TestSoftMul(0, 0xdeadbeef)

    def test_soft_mul_1(self):
        self.TestSoftMul(1, 0xdeadbeef)

    def test_soft_mul_2(self):
        self.TestSoftMul(2, 0xdeadbeef)



@unittest.skipIf(disableVerilatorTests, "Verilator")
class TestUncompressedOnCompressedIsa(TestNoCompression):
    hasCompressedIsa = True


@unittest.skipIf(disableVerilatorTests, "Verilator")
class TestCompressed(TestBase):
    hasCompressedIsa = True

    def test_c_li(self):

        testValue = 27
        offset = 0x14

        self.SetProgram([
            *Li(TEST_DATA_ADDR, 10),
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
            *Li(TEST_DATA_ADDR, 10),
            asm("C.LI", imm=testValue, rd=11),
            asm("C.ADDI", imm=2, rsd=11),
            asm("SW", imm=offset, rs1=10, rs2=11),
            asm("EBREAK")
        ])

        self.WaitEbreak()

        self.assertEqual(testValue + 2, self.mem.ReadWord(TEST_DATA_ADDR + offset))


def hash(x1, x2, seed):
    # h = seed
    c1 = 0xcc9e2d51
    # c2 = 0x1b873593

    # for x in [x1, x2]:
    #     k = x

    #     k *= c1
        #k = (k << 15) | (k >> (32 - 15))  # Rotate left 15 bits
        #k *= c2

        # h ^= k
        #h = (h << 13) | (h >> (32 - 13))  # Rotate left 13 bits
        #h = h * 5 + 0xe6546b64

    # Return as unsigned 32-bit integer
    # return h & 0xFFFFFFFF
    return (x1 * c1) & 0xFFFFFFFF


@unittest.skip("tmp")
@unittest.skipIf(disableVerilatorTests, "Verilator")
class TestWithFirmware(TestBase):
    baseDir = Path(__file__).parent

    def test_simple(self):
        with open(self.baseDir / "firmware" / "zig-out" / "bin" / "firmware_simple.bin", mode="rb") as f:
            fw = f.read()
        self.mem.WriteBytes(0, fw)

        testAddr = 0x800
        inValue = 0x12345678deadbeef
        testValue = hash(inValue & 0xFFFFFFFF, inValue >> 32, 0xc001babe)
        self.mem.WriteBytes(testAddr + 8, struct.pack("<Q", inValue))

        self.Reset()

        self.WaitEbreak()

        print(self.mem.ReadBytes(testAddr, 32))
        self.assertEqual(0, self.mem.ReadWord(testAddr))
        self.assertEqual(testValue, self.mem.ReadWord(testAddr + 4))
        self.assertEqual(b"1234567890", self.mem.ReadBytes(testAddr + 16, 10))



    # def test_print(self):
    #     with open(self.baseDir / "firmware" / "zig-out" / "bin" / "firmware_print.bin", mode="rb") as f:
    #         fw = f.read()
    #     self.mem.WriteBytes(0, fw)

    #     testValue = 0x12345678deadbeef
    #     testAddr = 0x800
    #     self.mem.WriteBytes(testAddr, struct.pack("<Q", testValue))

    #     self.Reset()

    #     self.WaitEbreak()

    #     # self.assertEqual(0, self.mem.ReadWord(testAddr + 8))
    #     addr = self.mem.ReadWord(testAddr + 8)
    #     print(addr)
    #     print(self.mem.ReadBytes(addr, 128))
    #     # print(self.mem.ReadBytes(testAddr, 128))


    def test_spigot(self):
        with open(self.baseDir / "firmware" / "zig-out" / "bin" / "firmware_spigot.bin", mode="rb") as f:
            fw = f.read()
        self.mem.WriteBytes(0, fw)

        testAddr = 0x800

        self.Reset()

        self.WaitEbreak()

        self.assertEqual(0, self.mem.ReadWord(testAddr))
        print(self.mem.ReadBytes(testAddr, 32))
