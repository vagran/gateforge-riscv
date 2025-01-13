# mypy: disable-error-code="type-arg, valid-type"
from pathlib import Path
import struct
import unittest

from gateforge.compiler import CompileModule
from gateforge.core import RenderOptions
from gateforge.dsl import always_comb, const, reg, wire
from gateforge.verilator import VerilatorParams
from riscv.instruction_decompressor import Assemble, Bindings, CommandTransform, Synthesize, \
    commands16
from test.utils import NullOutput, disableVerilatorTests


class TestBase(unittest.TestCase):
    def setUp(self):
        pass


    def tearDown(self):
        pass


class TestWithAssembler(TestBase):


    def test_basic(self):
        tmpObjFileName = "test_decompressor.o"

        for cmdName in commands16.keys():
            print(f"\n========================= {cmdName} =========================")
            cmd = commands16[cmdName]
            tcs = cmd.GenerateTestCases()
            for tc in tcs:
                print(f"[{cmd}] {tc}")
                asm = cmd.GenerateAsm(tc)
                print(asm)
                asmB = Assemble(asm, True, tmpObjFileName)
                opc = cmd.GenerateOpcode(tc)
                if asmB != opc:
                    raise Exception("Assembled opcode does not match the generated one: "  +
                                    f"{asmB.hex(' ')} vs {opc.hex(' ')}")

                # Compile base instruction
                assert cmd.mapTo is not None
                baseCmd = cmd.mapTo.targetCmd
                baseBindings = Bindings()
                baseBindings.Extend(tc)
                baseBindings.Extend(cmd.mapTo.bindings)
                print(baseBindings)
                asm = baseCmd.GenerateAsm(baseBindings)
                print(asm)
                asmB = Assemble(asm, True, tmpObjFileName)
                if asmB != opc:
                    raise Exception("Assembled base opcode does not match the generated one: "  +
                                    f"{asmB.hex(' ')} vs {opc.hex(' ')}")
                opc32 = baseCmd.GenerateOpcode(baseBindings)
                asmB = Assemble(asm, False, tmpObjFileName)
                if asmB != opc32:
                    raise Exception("Assembled full base opcode does not match the generated one: "  +
                                    f"{asmB.hex(' ')} vs {opc32.hex(' ')}")

                t = CommandTransform(cmd)
                decompressed = t.Apply(opc)
                if decompressed != opc32:
                    raise Exception(f"Bad decompressed value: {decompressed.hex(' ')} != {opc32.hex(' ')}")


def DecompressorTestbench():
    cmd16 = wire(16, "cmd16").input.port
    cmd32 = wire(32, "cmd32").output.port
    cmd30 = reg(30)
    with always_comb():
        Synthesize(cmd16, cmd30)
    cmd32 <<= cmd30 % const("2'b11")


@unittest.skipIf(disableVerilatorTests, "Verilator")
class TestWithVerilator(TestBase):
    def setUp(self):
        wspDir = Path(__file__).parent / "workspace"
        verilatorParams = VerilatorParams(buildDir=str(wspDir), quite=False)
        self.result = CompileModule(DecompressorTestbench, NullOutput(),
                                    renderOptions=RenderOptions(sourceMap=True),
                                    verilatorParams=verilatorParams)
        self.sim = self.result.simulationModel
        self.ports = self.sim.ports


    def tearDown(self):
        self.sim.Close()


    def test_decompression(self):
        for cmdName in commands16.keys():
            cmd = commands16[cmdName]
            tcs = cmd.GenerateTestCases()
            for tc in tcs:
                opc16 = cmd.GenerateOpcode(tc)
                assert cmd.mapTo is not None
                baseCmd = cmd.mapTo.targetCmd
                baseBindings = Bindings()
                baseBindings.Extend(tc)
                baseBindings.Extend(cmd.mapTo.bindings)
                opc32 = baseCmd.GenerateOpcode(baseBindings)
                self.ports.cmd16 = struct.unpack(">H", opc16)[0]
                self.sim.Eval()
                cmd32 = struct.unpack(">I", opc32)[0]
                try:
                    self.assertEqual(self.ports.cmd32, cmd32)
                except:
                    print(f"Failed command: {cmdName} {tc}")
                    raise
