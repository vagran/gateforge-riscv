import unittest
from riscv.instruction_decompressor import Assemble, Bindings, CommandTransform, commands16


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
