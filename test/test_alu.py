# mypy: disable-error-code="type-arg, valid-type"

import unittest

from gateforge.compiler import CompileModule
from gateforge.core import RenderOptions
from gateforge.dsl import wire
from riscv.alu import Alu
from test.utils import GetVerilatorParams, NullOutput, disableVerilatorTests


SIZE = 8


class Testbench:
    alu: Alu

    def __init__(self):
        self.alu = Alu(inA=wire("inA", SIZE).input.port,
                       inB=wire("inB", SIZE).input.port,
                       isSub=wire("isSub").input.port,
                       size=SIZE)


    def __call__(self):

        wire("outOr", SIZE).output.port <<= self.alu.outOr
        wire("outAnd", SIZE).output.port <<= self.alu.outAnd
        wire("outXor", SIZE).output.port <<= self.alu.outXor
        wire("outAddSub", SIZE).output.port <<= self.alu.outAddSub
        wire("outZ").output.port <<= self.alu.outZ
        wire("outLt").output.port <<= self.alu.outLt
        wire("outLtu").output.port <<= self.alu.outLtu

        self.alu()

        return self


    @staticmethod
    def AluModule():
        Testbench()()


def signExtend(value, size):
    signBit = 1 << (size - 1)
    return (value & (signBit - 1)) - (value & signBit)


class TestBase(unittest.TestCase):
    def setUp(self):
        self.result = CompileModule(Testbench.AluModule, NullOutput(),
                                    renderOptions=RenderOptions(sourceMap=True),
                                    verilatorParams=GetVerilatorParams())
        self.sim = self.result.simulationModel
        self.ports = self.sim.ports


    def tearDown(self):
        self.sim.Close()


    def Check(self, expected: int, actual: int):
        expected = expected & ((1 << SIZE) - 1)
        self.assertEqual(f"0x{actual:02x}", f"0x{expected:02x}")


    def Test(self, inA: int, inB: int):
        self.ports.inA = inA
        self.ports.inB = inB
        for isSub in [0, 1]:
            self.ports.isSub = isSub
            self.sim.Eval()
            self.Check(inA | inB, self.ports.outOr)
            self.Check(inA & inB, self.ports.outAnd)
            self.Check(inA ^ inB, self.ports.outXor)

        self.ports.isSub = 0
        self.sim.Eval()
        self.Check(inA + inB, self.ports.outAddSub)
        self.assertEqual(1 if (inA + inB) & 0xff == 0 else 0, self.ports.outZ)

        self.ports.isSub = 1
        self.sim.Eval()
        self.Check(inA - inB, self.ports.outAddSub)

        inAu = inA & 0xff
        inBu = inB & 0xff

        self.assertEqual(1 if inAu == inBu else 0, self.ports.outZ)
        self.assertEqual(1 if inAu < inBu else 0, self.ports.outLtu)

        inAs = signExtend(inAu, SIZE)
        inBs = signExtend(inBu, SIZE)

        self.assertEqual(1 if inAs < inBs else 0, self.ports.outLt)


@unittest.skipIf(disableVerilatorTests, "Verilator")
class Test(TestBase):

    def test_basic(self):
        self.Test(0, 0)
        self.Test(0xa5, 0)
        self.Test(0x0, 0xa5)
        self.Test(0xa5, 0xa5)
        self.Test(0xa5, 0x5a)
        self.Test(0x5a, 0xa5)
        self.Test(0xff, 0)
        self.Test(0, 0xff)
        self.Test(0xff, 0xff)
        self.Test(0x39, 0x75)
        self.Test(0x75, 0x39)
        self.Test(-0x35, 0x39)
        self.Test(0x39, -0x35)
        self.Test(-0x35, -0x35)
