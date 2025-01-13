# mypy: disable-error-code="type-arg, valid-type"
# This file contains declarative definition of compressed instructions set (see DefineCommands32()
# and DefineCommands16()), and corresponding code to generate hardware definition and test data from
# it.

import argparse
from enum import Enum, auto
import os
import re
import subprocess
from typing import Any, Optional, Sequence

from gateforge.core import Expression, InputNet, OutputNet, Reg, Wire
from gateforge.dsl import _else, _if, concat, const

args: Any = None


class OpcodeComponent:
    position: int

    """Bit-field in a command opcode.
    """
    def GetSize(self) -> int:
        """
        :return: Field size in bits
        """
        raise Exception("Method not implemented")


class Bindings:
    items: list[tuple[OpcodeComponent, int]]


    def __init__(self) -> None:
        # List of tuples (field reference, value)
        self.items = []


    def __str__(self) -> str:
        s = ""
        for b in self.items:
            if len(s) > 0:
                s += " "
            s += f"{b[0]}: {b[1]}"
        return s


    def Append(self, binding: tuple[OpcodeComponent, int]):
        self.items.append(binding)


    def Extend(self, bindings: "Bindings | list[tuple[OpcodeComponent, int]]"):
        if isinstance(bindings, Bindings):
            self.items.extend(bindings.items)
        else:
            self.items.extend(bindings)


    def Match(self, ref: OpcodeComponent) -> Optional[int]:
        """
        :param ref: Immediate or register reference.
        :return Binding value, None if not found.
        """
        if isinstance(ref, ImmediateBits):
            predicate = lambda ref: isinstance(ref, ImmediateBits)
        elif isinstance(ref, RegReference):
            if ref.regType == RegType.SRC1:
                regList = [RegType.SRC1, RegType.SRC_DST]
            elif ref.regType == RegType.DST:
                regList = [RegType.DST, RegType.SRC_DST]
            else:
                regList = [ref.regType]
            predicate = lambda c: isinstance(c, RegReference) and (c.regType in regList)
        else:
            raise Exception("Unsupported reference type")
        value = next((b[1] for b in self.items if predicate(b[0])), None)
        if isinstance(ref, RegReference) and ref.isNotEqual is not None and value == ref.isNotEqual:
            raise Exception("Constrained register matched to disallowed binding")
        return value


def BitStringToBytes(s: str) -> bytes:
    l = []
    idx = 0
    while idx < len(s):
        l.append(int(s[idx:idx+8], base=2))
        idx += 8
    return bytes(l)


class CommandDesc:
    """Command op-code definition.
    """

    name: str
    components: Sequence[OpcodeComponent]
    # Mapping to target 32-bit instruction. Specified only for compressed op-codes.
    mapTo: Optional["DecompressionMapping"]
    # Immediate in the op-code is an offset applied to register content.
    isImmOffset: bool
    immIsSigned: Optional[bool]
    immHiBit: Optional[int]


    def __init__(self, name, components: Sequence[OpcodeComponent],
                 mapTo: Optional["DecompressionMapping"] = None,
                 isImmOffset=False) -> None:
        self.name = name
        self.components = components
        self.mapTo = mapTo
        self.isImmOffset = isImmOffset

        self.immIsSigned = None
        self.immHiBit = None
        curPos = self.GetSize() - 1
        for c in components:
            c.position = curPos
            curPos -= c.GetSize()
            if isinstance(c, ImmediateBits):
                if self.immIsSigned is None:
                    self.immIsSigned = c.isSigned
                elif self.immIsSigned != c.isSigned:
                    raise Exception(f"Mixing signed and unsigned immediate field in one command: {name}")
                assert c.hiBit is not None
                if self.immHiBit is None or self.immHiBit < c.hiBit:
                    self.immHiBit = c.hiBit

        # Figure out immediate alignment if any (count missing LSB)
        self.immAlign = 0
        if self.immIsSigned is not None:
            for i in range(32):
                if self.FindImmediate(i) is None:
                    self.immAlign = i + 1
                else:
                    break


    def __str__(self) -> str:
        return self.name


    def VerifyBindings(self, bindings: Bindings):
        for b in bindings.items:
            param = self.FindParam(b[0])
            if param is None:
                raise Exception(
                    f"Cannot match binding of type {b[0].__class__.__name__} to command {self.name}")
            if isinstance(param, RegReference) and param.isNotEqual is not None and \
                b[1] == param.isNotEqual:
                raise Exception("Constrained register bound to disallowed value")


    def GetSize(self) -> int:
        size = 0
        for comp in self.components:
            size += comp.GetSize()
        return size


    def FindParam(self, paramType: OpcodeComponent) -> Optional[OpcodeComponent]:

        if isinstance(paramType, imm):
            predicate = lambda c: isinstance(c, imm)

        elif isinstance(paramType, RegReference):
            if paramType.regType == RegType.SRC1:
                regList = [RegType.SRC1, RegType.SRC_DST]
            elif paramType.regType == RegType.DST:
                regList = [RegType.DST, RegType.SRC_DST]
            else:
                regList = [paramType.regType]
            predicate = lambda c: isinstance(c, RegReference) and (c.regType in regList)

        else:
            raise Exception(f"Unsupported parameter type: {paramType}")

        return next((c for c in self.components if predicate(c)), None)


    def FindImmediate(self, immBit: int) -> Optional["ImmediateBits"]:
        """
        :param immBit: Immediate bit index to find.
        :return: Immediate chunk reference with the specified bit if found, None if not found.
        """
        for c in self.components:
            if not isinstance(c, ImmediateBits):
                continue
            assert c.hiBit is not None
            assert c.loBit is not None
            if immBit <= c.hiBit and immBit >= c.loBit:
                return c
        return None


    def GetField(self, bitIdx: int) -> Optional[OpcodeComponent]:
        """_summary_
        :return: Field containing the specified bit.
        """
        for c in self.components:
            if bitIdx > c.position:
                raise Exception("Bit index out of range")
            if bitIdx <= c.position and bitIdx >= c.position - c.GetSize() + 1:
                return c
        raise Exception("Failed to find field")


    def GetConstrainedRegisterFields(self) -> list["RegReference"]:
        """
        :return: List of register reference fields with not-equal value.
        """
        l = []
        for c in self.components:
            if isinstance(c, RegReference) and c.isNotEqual is not None:
                l.append(c)
        return l


    def GenerateTestCases(self) -> list[Bindings]:
        """
        :return List of lists of bindings for test cases for this command.
        """
        result = []

        def Generate(positiveImm) -> Bindings:
            bindings = Bindings()
            if self.immIsSigned is not None:
                if self.immAlign == 0:
                    v = 10
                else:
                    v = 3 << self.immAlign
                bindings.Append((imm(), v if positiveImm else -v))
            # Use x10 and above
            curReg = 10
            for c in self.components:
                if not isinstance(c, RegReference):
                    continue
                bindings.Append((c, curReg))
                curReg += 1
            return bindings

        result.append(Generate(True))
        if self.immIsSigned:
            # Case with negative immediate value if has one signed
            result.append(Generate(False))
        return result


    def GenerateOpcode(self, bindings: Bindings) -> bytes:
        """
        :return: Opcode bytes.
        """

        # First generate bit string, MSB to LSB
        s = ""

        for c in self.components:
            if isinstance(c, ConstantBits):
                s += c.GetBitString()

            elif isinstance(c, RegReference):
                value = bindings.Match(c)
                if value is None:
                    raise Exception("Failed to match reg ref against provided bindings")
                s += c.BindValue(value).GetBitString()

            elif isinstance(c, ImmediateBits):
                value = bindings.Match(c)
                if value is None:
                    raise Exception("Failed to match immediate against provided bindings")
                if value < 0 and not c.isSigned:
                    raise Exception("Negative value bound for unsigned immediate")
                assert c.hiBit is not None
                assert c.loBit is not None
                s += ConstantBits.FromInt(32, value).Slice(c.hiBit, c.loBit).GetBitString()

            else:
                raise Exception(f"Unrecognized field: {c}")

        if len(s) != 16 and len(s) != 32:
            raise Exception(f"Bad opcode length: {len(s)}")

        return BitStringToBytes(s)


    def GenerateAsm(self, bindings: Bindings) -> str:
        """
        :param bindings: Bindings to use for arguments.
        :return: Assembler text for the command.
        """
        asm = self.name
        # Destination register if any
        _rd: RegReference = self.FindParam(rd()) # type: ignore
        if _rd is not None:
            v = bindings.Match(_rd)
            if v is None:
                raise Exception("Failed to match destination register against bindings")
            asm += f" x{v}"

        _rs1 = self.FindParam(rs1())
        _rs2 = self.FindParam(rs2())
        _imm = self.FindParam(imm())

        if (_rs1 is not None and (_rd is None or _rd.regType != RegType.SRC_DST) and
            not self.isImmOffset):

            v = bindings.Match(_rs1)
            if v is None:
                raise Exception("Failed to match source register 1 against bindings")
            asm += " " if _rd is None else ", "
            asm += f"x{v}"

        # Workaround for assembler bug which requires x2 to be always specified for C.ADDI4SPN and
        # C.ADDI16SP.
        if self.name == "C.ADDI4SPN" or self.name == "C.ADDI16SP":
            asm += " " if _rd is None else ", "
            asm += "x2"
            _rs1 = rs1()

        if _rs2 is not None:
            v = bindings.Match(_rs2)
            if v is None:
                raise Exception("Failed to match source register 2 against bindings")
            asm += " " if _rd is None and self.isImmOffset else ", "
            asm += f"x{v}"

        if _imm is not None:
            v = bindings.Match(imm())
            if v is None:
                raise Exception("Failed to match immediate value against bindings")

            # C.LUI command expects scaled immediate and negative value as unsigned
            if self.name == "C.LUI" or self.name == "LUI":
                v = v >> 12
                if v < 0:
                    v = 0x100000 + v

            if self.isImmOffset:
                vrs1: Optional[int]
                if self.name == "C.LWSP" or self.name == "C.SWSP":
                    # x2 for C.LWSP
                    vrs1 = 2
                else:
                    if _rs1 is None:
                        raise Exception("Offset immediate without source register 1")
                    vrs1 = bindings.Match(_rs1)
                    if vrs1 is None:
                        raise Exception("Failed to match source register 1 against bindings")
                asm += " " if _rd is None and _rs2 is None else ", "
                asm += f"{v}(x{vrs1})"
            else:
                asm += " " if _rd is None and _rs1 is None else ", "
                asm += f"{v}"

        return asm


    def GetConstantBit(self, bitIdx: int) -> Optional[int]:
        """
        :param bitIdx: Bit index to test.
        :return: Value of constant bit (1 or 0), or None if the bit is not constant.
        """
        for c in self.components:
            csz = c.GetSize()
            if bitIdx > c.position:
                raise Exception("Bit index out of range")
            if bitIdx < c.position - csz + 1:
                continue
            if isinstance(c, ConstantBits):
                return 1 if (c.value & (1 << (c.size - 1 - c.position + bitIdx))) != 0 else 0
            return None
        raise Exception("Unexpected end of list")


# Indexed by command name
commands32: dict[str,  CommandDesc] = {}
commands16: dict[str,  CommandDesc] = {}

# ##################################################################################################
# Elements for declarative description of commands

class ConstantBits(OpcodeComponent):
    size: int
    value: int

    """
    Some constant bits
    """
    def __init__(self, bits):
        """
        :param bits: String with binary representation of constant bits. Length should correspond to
        number of bits (do not skip leading zeros).
        """
        super().__init__()
        self.size = len(bits)
        if self.size > 32:
            raise Exception("Too long field")
        self.value = int(bits, base=2)


    @staticmethod
    def FromInt(size: int, value: int) -> "ConstantBits":
        max = (1 << size) - 1
        if value > max or value < -(max + 1) / 2:
            raise Exception(f"Value out of range: {value}")
        if value < 0:
            value = max + 1 + value
        return ConstantBits(format(value, f"0{size}b"))


    def GetSize(self) -> int:
        return self.size


    def GetBitString(self) -> str:
        return format(self.value, f"0{self.size}b")


    def __str__(self) -> str:
        return self.GetBitString()


    def Slice(self, hiBit: int, loBit: int) -> "ConstantBits":
        """Get subset of this bits chunk.
        :param hiBit: High-order bit of the slice.
        :param loBit: Low-order bit of the slice.
        :return: _description_
        """
        if hiBit > self.size - 1:
            raise Exception(f"hiBit out of range: {hiBit}")
        if loBit > hiBit:
            raise Exception(f"loBit is greater than hiBit: {loBit}")
        return ConstantBits(self.GetBitString()[self.size - 1 - hiBit : self.size - loBit])


b = ConstantBits

class ImmediateBits(OpcodeComponent):
    hiBit: Optional[int]
    loBit: Optional[int]
    isSigned: bool

    """
    Some chunk of immediate value.
    """
    def __init__(self, hiBit: Optional[int] = None, loBit: Optional[int] = None,
                 isSigned: bool = True):
        """
        :param hiBit: Index of high-ordered bit of the chunk. Can be None when used as bind target.
        :param loBit: Index of low-ordered bit of the chunk, None if one bit chunk.
        """
        super().__init__()
        if loBit is not None and hiBit is not None and loBit > hiBit:
            raise Exception("loBit is greater than hiBit")
        self.hiBit = hiBit
        if loBit is None:
            self.loBit = hiBit
        else:
            self.loBit = loBit
        self.isSigned = isSigned


    def GetSize(self) -> int:
        if self.hiBit is None:
            raise Exception("No size for bind target")
        return 1 if self.loBit is None else self.hiBit - self.loBit + 1


    def __str__(self) -> str:
        s = ""
        if not self.isSigned:
            s += "u"
        s += "imm"
        if self.hiBit is not None:
            s += "["
            s += str(self.hiBit)
            if self.loBit != self.hiBit:
                s += ":"
                s += str(self.loBit)
            s += "]"
        return s


imm = ImmediateBits

def uimm(hiBit: int, loBit: Optional[int] = None) -> ImmediateBits:
    """
    Chunk of unsigned immediate value.
    """
    return imm(hiBit, loBit, False)


class RegType(Enum):
    SRC1 = auto()
    SRC2 = auto()
    DST = auto()
    SRC_DST = auto()


class RegReference(OpcodeComponent):
    regType: RegType
    isCompressed: bool
    # Field is defined for registers except this one if specified
    isNotEqual: Optional[int]


    def __init__(self, regType: RegType, isCompressed=False, isNotEqual: Optional[int] =None):
        super().__init__()
        self.regType = regType
        self.isCompressed = isCompressed
        self.isNotEqual = isNotEqual


    def GetSize(self):
        return 3 if self.isCompressed else 5


    def BindValue(self, value):
        """
        :param value: value Register index
        :return ConstantBits for the field with the specified value.
        """
        if value < 0 or value > 15:
            raise Exception(f"Illegal register index: {value}")
        if self.isCompressed:
            if value < 8:
                raise Exception(f"Illegal register index for compressed field: {value}")
            return ConstantBits.FromInt(3, value - 8)
        else:
            return ConstantBits.FromInt(5, value)


    def __str__(self) -> str:
        s = "r"
        if self.regType == RegType.SRC1:
            s += "s1"
        elif self.regType == RegType.SRC2:
            s += "s2"
        elif self.regType == RegType.DST:
            s += "d"
        elif self.regType == RegType.SRC_DST:
            s += "d/rs1"
        if self.isCompressed:
            s += "'"
        return s


def rs1() -> RegReference:
    return RegReference(RegType.SRC1)

def rs2(isNotEqual: Optional[int] = None) -> RegReference:
    return RegReference(RegType.SRC2, isNotEqual=isNotEqual)

def rd(isNotEqual: Optional[int] = None) -> RegReference:
    return RegReference(RegType.DST, isNotEqual=isNotEqual)

def rsd():
    return RegReference(RegType.SRC_DST)

# rs1' (prime) - compressed representation
def rs1p() -> RegReference:
    return RegReference(RegType.SRC1, True)

def rs2p() -> RegReference:
    return RegReference(RegType.SRC2, True)

def rdp() -> RegReference:
    return RegReference(RegType.DST, True)

def rsdp() -> RegReference:
    return RegReference(RegType.SRC_DST, True)


def cmd32(name: str, *components: OpcodeComponent, isImmOffset=False):
    if name in commands32:
        raise Exception(f"Command {name} already defined")
    cmd = CommandDesc(name, components, isImmOffset=isImmOffset)
    if cmd.GetSize() != 32:
        raise Exception(f"Command size is not 32 bits: {cmd.GetSize()} for {name}")
    commands32[name] = cmd


def cmd16(name: str, mapTo: "DecompressionMapping", *components: OpcodeComponent, isImmOffset=False):
    if name in commands16:
        raise Exception(f"Command {name} already defined")
    cmd = CommandDesc(name, components, mapTo=mapTo, isImmOffset=isImmOffset)
    if cmd.GetSize() != 16:
        raise Exception(f"Command size is not 16 bits: {cmd.GetSize()} for {name}")
    commands16[name] = cmd


class DecompressionMapping:
    targetCmd: CommandDesc
    bindings: Bindings

    def __init__(self, cmdName: str, bindings: Optional[Bindings] = None):
        """Mapping to 32 bits command.
        :param cmdName: 32 bits command name
        :param bindings: list of tuples (field, value)
        """
        if cmdName not in commands32:
            raise Exception(f"Target command {cmdName} not found")
        self.targetCmd = commands32[cmdName]
        self.bindings = Bindings()
        if bindings is not None:
            self.bindings.Extend(bindings)
            self.targetCmd.VerifyBindings(self.bindings)


    def FindBinding(self, component: OpcodeComponent) -> Optional[int]:
        """
        :param component: Immediate or register reference.
        :return Binding value, None if not found.
        """
        return self.bindings.Match(component)


mapTo = DecompressionMapping

# ##################################################################################################
# Commands declarative description based on RISC-V Instruction Set Manual

def DefineCommands32():
    cmd = cmd32

    cmd("LW",
        imm(11,0), rs1(), b("010"), rd(), b("0000011"),
        isImmOffset=True)
    cmd("SW",
        imm(11,5), rs2(), rs1(), b("010"), imm(4,0), b("0100011"),
        isImmOffset=True)
    cmd("JAL",
        imm(20), imm(10,1), imm(11), imm(19,12), rd(), b("1101111"))
    cmd("JALR",
        imm(11,0), rs1(), b("000"), rd(), b("1100111"))
    cmd("BEQ",
        imm(12), imm(10,5), rs2(), rs1(), b("000"), imm(4,1), imm(11), b("1100011"))
    cmd("BNE",
        imm(12), imm(10,5), rs2(), rs1(), b("001"), imm(4,1), imm(11), b("1100011"))
    cmd("ADDI",
        imm(11,0), rs1(), b("000"), rd(), b("0010011"))
    cmd("LUI",
        imm(31,12), rd(), b("0110111"))
    cmd("SLLI",
        b("0000000"), uimm(4,0), rs1(), b("001"), rd(), b("0010011"))
    cmd("SRLI",
        b("0000000"), uimm(4,0), rs1(), b("101"), rd(), b("0010011"))
    cmd("SRAI",
        b("0100000"), uimm(4,0), rs1(), b("101"), rd(), b("0010011"))
    cmd("ANDI",
        imm(11,0), rs1(), b("111"), rd(), b("0010011"))
    cmd("ADD",
        b("0000000"), rs2(), rs1(), b("000"), rd(), b("0110011"))
    cmd("SUB",
        b("0100000"), rs2(), rs1(), b("000"), rd(), b("0110011"))
    cmd("XOR",
        b("0000000"), rs2(), rs1(), b("100"), rd(), b("0110011"))
    cmd("OR",
        b("0000000"), rs2(), rs1(), b("110"), rd(), b("0110011"))
    cmd("AND",
        b("0000000"), rs2(), rs1(), b("111"), rd(), b("0110011"))


def DefineCommands16():
    cmd = cmd16

    cmd("C.ADDI4SPN", mapTo("ADDI", [(rs1(), 2)]),
        b("000"), uimm(5,4), uimm(9,6), uimm(2), uimm(3), rdp(), b("00"))
    cmd("C.LW", mapTo("LW"),
        b("010"), uimm(5,3), rs1p(), uimm(2), uimm(6), rdp(), b("00"),
        isImmOffset=True)
    cmd("C.SW", mapTo("SW"),
        b("110"), uimm(5,3), rs1p(), uimm(2), uimm(6), rs2p(), b("00"),
        isImmOffset=True)

    # No special handling for C.NOP - translate it to `ADDI x0, x0, 0` to save resources
    cmd("C.ADDI", mapTo("ADDI"),
        b("000"), imm(5), rsd(), imm(4,0), b("01"))
    cmd("C.JAL", mapTo("JAL", [(rd(), 1)]),
        b("001"), imm(11), imm(4), imm(9,8), imm(10), imm(6), imm(7), imm(3,1), imm(5), b("01"))
    cmd("C.LI", mapTo("ADDI", [(rs1(), 0)]),
        b("010"), imm(5), rd(), imm(4,0), b("01"))
    cmd("C.ADDI16SP", mapTo("ADDI", [(rs1(), 2), (rd(), 2)]),
        b("011"), imm(9), b("00010"), imm(4), imm(6), imm(8,7), imm(5), b("01"))
    cmd("C.LUI", mapTo("LUI"),
        b("011"), imm(17), rd(2), imm(16,12), b("01"))
    cmd("C.SRLI", mapTo("SRLI"),
        b("100"), b("0"), b("00"), rsdp(), uimm(4,0), b("01"))
    cmd("C.SRAI", mapTo("SRAI"),
        b("100"), b("0"), b("01"), rsdp(), uimm(4,0), b("01"))
    cmd("C.ANDI", mapTo("ANDI"),
        b("100"), imm(5), b("10"), rsdp(), imm(4,0), b("01"))
    cmd("C.SUB", mapTo("SUB"),
        b("100"), b("0"), b("11"), rsdp(), b("00"), rs2p(), b("01"))
    cmd("C.XOR", mapTo("XOR"),
        b("100"), b("0"), b("11"), rsdp(), b("01"), rs2p(), b("01"))
    cmd("C.OR", mapTo("OR"),
        b("100"), b("0"), b("11"), rsdp(), b("10"), rs2p(), b("01"))
    cmd("C.AND", mapTo("AND"),
        b("100"), b("0"), b("11"), rsdp(), b("11"), rs2p(), b("01"))
    cmd("C.J", mapTo("JAL", [(rd(), 0)]),
        b("101"), imm(11), imm(4), imm(9,8), imm(10), imm(6), imm(7), imm(3,1), imm(5), b("01"))
    cmd("C.BEQZ", mapTo("BEQ", [(rs2(), 0)]),
        b("110"), imm(8), imm(4,3), rs1p(), imm(7,6), imm(2,1), imm(5), b("01"))
    cmd("C.BNEZ", mapTo("BNE", [(rs2(), 0)]),
        b("111"), imm(8), imm(4,3), rs1p(), imm(7,6), imm(2,1), imm(5), b("01"))

    cmd("C.SLLI", mapTo("SLLI"),
        b("000"), b("0"), rsd(), uimm(4,0), b("10"))
    cmd("C.LWSP", mapTo("LW", [(rs1(), 2)]),
        b("010"), uimm(5), rd(), uimm(4,2), uimm(7,6), b("10"),
        isImmOffset=True)
    cmd("C.JR", mapTo("JALR", [(rd(), 0), (imm(), 0)]),
        b("100"), b("0"), rs1(), b("00000"), b("10"))
    cmd("C.MV", mapTo("ADD", [(rs1(), 0)]),
        b("100"), b("0"), rd(), rs2(0), b("10"))
    # C.EBREAK is skipped intentionally to save resources
    cmd("C.JALR", mapTo("JALR", [(rd(), 1), (imm(), 0)]),
        b("100"), b("1"), rs1(), b("00000"), b("10"))
    cmd("C.ADD", mapTo("ADD"),
        b("100"), b("1"), rsd(), rs2(0), b("10"))
    cmd("C.SWSP", mapTo("SW", [(rs1(), 2)]),
        b("110"), uimm(5,2), uimm(7,6), rs2(), b("10"),
        isImmOffset=True)


# ##################################################################################################

class BitsCopy:
    srcLo: int
    srcHi: int

    def __init__(self, srcHi: int, srcLo: Optional[int], numReplicate: Optional[int] = None):
        """Bits chunk copying operation.
        :param srcHi: Index of source high-ordered bit.
        :param srcLo: Index of source low-ordered bit. May be None to indicate one bit size.
        :numReplicate: Source bit (source must be one bit size) is replicated so many times if
        specified (used for sign extension).
        """
        self.srcHi = srcHi
        if srcLo is None:
            self.srcLo = srcHi
        else:
            if srcLo > srcHi:
                raise Exception("srcLo is greater than srcHi")
            self.srcLo = srcLo
        if numReplicate is not None:
            if self.srcLo != self.srcHi:
                raise Exception("Replication count can be specified for 1-bit source only")
        self.numReplicate = numReplicate


    def CopyFromBitString(self, s: str) -> str:
        """
        :param s: Source bit string.
        :return: Slice with corresponding bits.
        """
        if len(s) != 16:
            raise Exception("Expected 16 bits bit-string")
        if self.numReplicate is not None:
            return s[15 - self.srcHi] * self.numReplicate
        return s[15 - self.srcHi : 16 - self.srcLo]


class CommandTransform:
    srcCmd: CommandDesc
    components: list[ConstantBits | BitsCopy]


    """Implements command transformation from compressed 16-bits representation to 32-bits
    representation.
    """
    def __init__(self, cmd16Desc: CommandDesc):
        self.srcCmd = cmd16Desc
        self.components = []
        assert cmd16Desc.mapTo is not None
        targetCmd = cmd16Desc.mapTo.targetCmd
        for c in targetCmd.components:
            if isinstance(c, ConstantBits):
                self.components.append(c)

            elif isinstance(c, RegReference):
                binding = cmd16Desc.mapTo.FindBinding(c)
                if binding is not None:
                    self.components.append(ConstantBits.FromInt(5, binding))
                else:
                    regRef: RegReference = cmd16Desc.FindParam(c) # type: ignore
                    if regRef is None:
                        raise Exception("Cannot find register in source " +
                                        f"command: {c.regType} in {cmd16Desc.name}")
                    if regRef.isCompressed:
                        # Register field in target is always 5 bits
                        self.components.append(ConstantBits("01"))
                        self.components.append(BitsCopy(regRef.position, regRef.position - 2))
                    else:
                        self.components.append(BitsCopy(regRef.position, regRef.position - 4))

            elif isinstance(c, ImmediateBits):
                binding = cmd16Desc.mapTo.FindBinding(c)
                if binding is not None:
                    assert targetCmd.immHiBit is not None
                    bits = ConstantBits.FromInt(targetCmd.immHiBit + 1, binding)
                    assert c.hiBit is not None
                    assert c.loBit is not None
                    self.components.append(bits.Slice(c.hiBit, c.loBit))
                else:
                    self._HandleImmediateChunk(c)

            else:
                raise Exception(f"Unhandled component type {c}")

        self._FoldReplications()
        self._FoldConstantBits()


    def _HandleImmediateChunk(self, c: ImmediateBits):
        if self.srcCmd.immHiBit is None:
            raise Exception("No immediate field in source command")

        isZeroBits = None
        hiBit = None
        loBit = None
        srcHiBit = None

        assert c.hiBit is not None
        assert c.loBit is not None
        assert self.srcCmd.immHiBit is not None

        def CommitBits():
            nonlocal isZeroBits, hiBit, loBit, srcHiBit

            if isZeroBits is None:
                return
            if isZeroBits:
                self.components.append(ConstantBits.FromInt(hiBit - loBit + 1, 0))
            else:
                self.components.append(BitsCopy(srcHiBit, srcHiBit - (hiBit - loBit)))
            isZeroBits = None
            hiBit = None
            loBit = None
            srcHiBit = None

        def CopyBit(immBit):
            nonlocal isZeroBits, hiBit, loBit, srcHiBit

            srcImm = self.srcCmd.FindImmediate(immBit)
            if isZeroBits is not None and isZeroBits != srcImm is None:
                CommitBits()
            if isZeroBits is None:
                if srcImm is None:
                    isZeroBits = True
                else:
                    isZeroBits = False
                    srcHiBit = srcImm.position - (srcImm.hiBit - immBit)
                hiBit = immBit
            elif not isZeroBits:
                srcBit = srcImm.position - (srcImm.hiBit - immBit)
                if srcBit != srcHiBit - (srcImm.hiBit - immBit):
                    # Discontinuity in bits chunk, start new chunk
                    CommitBits()
                    isZeroBits = False
                    srcHiBit = srcImm.position - (srcImm.hiBit - immBit)
                    hiBit = immBit
            loBit = immBit

        if c.hiBit > self.srcCmd.immHiBit:
            # Handle sign bits
            loSignBit = max(c.loBit, self.srcCmd.immHiBit + 1)
            if self.srcCmd.immIsSigned:
                # Sign extension, find chunk with sign bit in source command
                srcImm = self.srcCmd.FindImmediate(self.srcCmd.immHiBit)
                if srcImm is None:
                    raise Exception("Sign bit not found")
                self.components.append(BitsCopy(srcImm.position, None, c.hiBit - loSignBit + 1))
            else:
                # Leading zeros
                self.components.append(ConstantBits.FromInt(c.hiBit - loSignBit + 1, 0))
            # Handle rest bits of chunk if any
            for immBit in range(loSignBit - 1, c.loBit - 1, -1):
                CopyBit(immBit)
        else:
            # Just copy all bits (missing ones are set to zero)
            for immBit in range(c.hiBit, c.loBit - 1, -1):
                CopyBit(immBit)

        CommitBits()


    def _FoldReplications(self):
        # Fold several adjacent bits replications into one component
        out = []
        s = None

        def Commit():
            nonlocal s
            if s is None:
                return
            out.append(ConstantBits(s))
            s = None

        def Add(c):
            nonlocal s
            if s is None:
                s = c.GetBitString()
            else:
                s += c.GetBitString()

        for c in self.components:
            if isinstance(c, ConstantBits):
                Add(c)
            else:
                Commit()
                out.append(c)

        Commit()
        self.components = out


    def _FoldConstantBits(self):
        # Fold several adjacent constant bits into one component
        out = []
        srcBit = None
        count = 0

        def Commit():
            nonlocal srcBit, count
            if srcBit is None:
                return
            out.append(BitsCopy(srcBit, None, count if count > 1 else None))
            srcBit = None
            count = 0

        def Add(c):
            nonlocal srcBit, count
            if srcBit is not None and c.srcHi != srcBit:
                Commit()
            srcBit = c.srcHi
            count += c.numReplicate if c.numReplicate is not None else 1

        for c in self.components:
            if isinstance(c, BitsCopy) and c.srcLo == c.srcHi:
                Add(c)
            else:
                Commit()
                out.append(c)

        Commit()
        self.components = out


    def Apply(self, opcode16: bytes) -> bytes:
        """
        :param opcode16: 16-bits opcode (bytes) to apply transform on.
        :return 32-bits decompressed opcode (bytes).
        """
        s = ""
        opcode16String = ConstantBits.FromInt(16, opcode16[0] << 8 | opcode16[1]).GetBitString()
        for c in self.components:
            if isinstance(c, ConstantBits):
                s += c.GetBitString()
            elif isinstance(c, BitsCopy):
                s += c.CopyFromBitString(opcode16String)
            else:
                raise Exception("Bad component type")
        if len(s) != 32:
            raise Exception(f"Unexpected result size: {len(s)}")
        return BitStringToBytes(s)


    def SynthesizeExpression(self, input: InputNet[Wire, 16]) -> Expression:
        result: list[Expression] = []
        lastComp = self.components[len(self.components) - 1]
        if not isinstance(lastComp, ConstantBits):
            raise Exception("Expected constant bits in last component")
        trimmedLastComp = lastComp.Slice(lastComp.size - 1, 2)
        for c in self.components:
            if c is lastComp:
                c = trimmedLastComp
            if isinstance(c, ConstantBits):
                result.append(const(c.value, c.size))
            elif isinstance(c, BitsCopy):
                if c.numReplicate is None:
                    if c.srcLo == c.srcHi:
                        result.append(input[c.srcHi])
                    else:
                        result.append(input[c.srcHi:c.srcLo])
                else:
                    result.append(input[c.srcHi].replicate(c.numReplicate))

            else:
                raise Exception("Unexpected component type")
        return concat(*result)


def Assemble(commandText: str, isCompressed: bool, tmpObjFileName: Optional[str] = None) -> bytes:
    """Assemble instruction into op-codes. Used for testing.
    :param commandText: Command test in assembler language.
    :param embeddedTarget: True to target RV32EC, false for RV32E.
    :return bytes for the command.
    """


    code = f"""
.text
{commandText}
    """
    compiler = os.environ["GF_TEST_CC"]
    objdump = os.environ["GF_TEST_OBJDUMP"]
    objFileName = tmpObjFileName if tmpObjFileName is not None else "/tmp/decompressor_test.o"
    subprocess.run([compiler, "-c", "--target=riscv32",
                    "-march=rv32e" + ("c" if isCompressed else ""),
                    "-mno-relax", "-mlittle-endian", "-x", "assembler", "-o", objFileName, "-"],
                   input=code.encode("UTF-8"), check=True)

    p = subprocess.run([objdump, "--disassemble", objFileName],
                       check=True, capture_output=True)

    output = p.stdout.decode("utf-8")
    pat = re.compile(r"^\s*\d+:\s+((?:[a-f0-9]{2}\s)+).*$")
    try:
        for line in output.splitlines():
            m = pat.fullmatch(line)
            if m is None:
                continue
            print(line)
            return bytes(reversed([int(h, base=16) for h in m.group(1).split()]))
        raise Exception("Failed to find compiled opcodes")
    finally:
        os.remove(objFileName)


class SelectionTree:
    """Contains logic for identifying input 16-bits command.
    """
    rootNode: "Node"


    class Node:
        # Temporal node (while building the tree) contains list of commands here, final interim node
        # contains Node instance, leaf node contains single command descriptor
        first: "Optional[list[CommandDesc] | CommandDesc | SelectionTree.Node]"
        second: "Optional[list[CommandDesc] | CommandDesc | SelectionTree.Node]"
        hiBit: int
        loBit: int
        notEqualValue: Optional[int]


        def __init__(self, hiBit, loBit=None, notEqualValue=None) -> None:
            # True case for if, either Node or CommandDesc
            self.first = None
            # else case for if
            self.second = None
            self.hiBit = hiBit
            self.loBit = loBit if loBit is not None else hiBit
            self.notEqualValue = notEqualValue


        def HasBetterBalance(self, node: "SelectionTree.Node") -> bool:
            """
            :return: True if this temporal node has better balance than the specified one.
            """
            assert(isinstance(self.first, list))
            assert(isinstance(self.second, list))
            assert(isinstance(node.first, list))
            assert(isinstance(node.second, list))
            return abs(len(self.first) - len(self.second)) < abs(len(node.first) - len(node.second))


        def GetConditionExpr(self, cmd16net: InputNet[Wire, 16]) -> Expression:
            if self.loBit == self.hiBit:
                return cmd16net[self.hiBit]
            return cmd16net[self.hiBit:self.loBit] != self.notEqualValue


    def __init__(self, rootNode: Node) -> None:
        self.rootNode = rootNode


    @staticmethod
    def Generate(commands) -> "SelectionTree":
        assert len(commands) > 1
        return SelectionTree(SelectionTree.GenerateNode(commands)) # type: ignore


    @staticmethod
    def GenerateNode(commands: Sequence[CommandDesc]) -> Node | CommandDesc:
        if len(commands) == 1:
            return commands[0]
        # List of possible nodes, one be selected with the best balance
        candidate = None
        for i in range(16):
            node = SelectionTree.TryGenerateNode(commands, i)
            if node is not None:
                if candidate is None or node.HasBetterBalance(candidate):
                    candidate = node
                assert isinstance(candidate.first, list)
                assert isinstance(candidate.second, list)
                if abs(len(candidate.first) - len(candidate.second)) < 2:
                    # Already optimal balance
                    break

        checkedPositions = set()
        # Check for constrained register fields
        for cmd in commands:
            fields = cmd.GetConstrainedRegisterFields()
            for field in fields:
                if field.position in checkedPositions:
                    continue
                checkedPositions.add(field.position)
                node = SelectionTree.TryGenerateNode(commands, field.position,
                                                     field.position - field.GetSize() + 1)
                if node is not None:
                    if candidate is None or node.HasBetterBalance(candidate):
                        candidate = node
                    assert isinstance(candidate.first, list)
                    assert isinstance(candidate.second, list)
                    if abs(len(candidate.first) - len(candidate.second)) < 2:
                        # Already optimal balance
                        break
            else:
                continue
            break

        if candidate is None:
            raise Exception("Failed to generate selector for nodes: " + ", ".join(map(str, commands)))
        assert isinstance(candidate.first, list)
        assert isinstance(candidate.second, list)
        candidate.first = SelectionTree.GenerateNode(candidate.first)
        candidate.second = SelectionTree.GenerateNode(candidate.second)
        return candidate


    @staticmethod
    def TryGenerateNode(commands: Sequence[CommandDesc], hiBit: int,
                        loBit: Optional[int] = None) -> Optional[Node]:
        """
        :param hiBit: MSB to test.
        :param loBit: LSB to test, None if one bit.
        :return: Temporal node (children are lists of commands), or None if cannot be created on the
        specified bits.
        """
        nzCommands: list[CommandDesc] = []
        zCommands: list[CommandDesc] = []
        notEqualValue: Optional[int] = None

        if loBit is None:
            # Single bit test
            for cmd in commands:
                bit = cmd.GetConstantBit(hiBit)
                if bit is None:
                    return None
                if bit == 0:
                    zCommands.append(cmd)
                else:
                    nzCommands.append(cmd)
        else:
            # Bit-field test against not-equal value.
            # Bit index to value
            value: dict[int, int] = {}
            for cmd in commands:
                isConstant = None
                for bitIdx in range(loBit, hiBit + 1):
                    bit = cmd.GetConstantBit(bitIdx)
                    if bit is None:
                        if isConstant is None:
                            isConstant = False
                        elif isConstant:
                            return None
                    else:
                        if isConstant is None:
                            isConstant = True
                        elif not isConstant:
                            return None
                        if bitIdx in value:
                            if value[bitIdx] != bit:
                                return None
                        else:
                            value[bitIdx] = bit
                if isConstant:
                    zCommands.append(cmd)
                else:
                    nzCommands.append(cmd)
            notEqualValue = 0
            for bitIdx in range(loBit, hiBit + 1):
                if value[bitIdx]:
                    notEqualValue = notEqualValue | (1 << (bitIdx - loBit))
            # Check if all nzCommands have constrained field in target bits
            for cmd in nzCommands:
                field = cmd.GetField(hiBit)
                if not isinstance(field, RegReference):
                    return None
                if field.position != hiBit or field.position - field.GetSize() + 1 != loBit:
                    return None
                if field.isNotEqual != notEqualValue:
                    return None

        if len(zCommands) == 0 or len(nzCommands) == 0:
            return None

        node = SelectionTree.Node(hiBit, loBit, notEqualValue)
        node.first = nzCommands
        node.second = zCommands
        return node


    def Synthesize(self, input: InputNet[Wire, 16], output: OutputNet[Reg, 32]):
        """
        Synthesizes decompression logic. Should be called in context of combinational procedural
        block.
        :param input: Input net with compressed 16-bits opcode.
        :param output: Output net to assign 32-bits decompressed opcode.
        """
        self._SynthesizeNode(self.rootNode, input, output)


    def _SynthesizeNode(self, node: "Node", insn16: InputNet[Wire, 16], insn32: OutputNet[Reg, 32]):

        with _if(node.GetConditionExpr(insn16)):

            if isinstance(node.first, CommandDesc):
                assert node.first.mapTo is not None
                # comment(f"{node.first} -> {node.first.mapTo.targetCmd}")
                t = CommandTransform(node.first)
                expr = t.SynthesizeExpression(insn16)
                if insn32.isReg:
                    insn32 <<= expr
                else:
                    insn32 //= expr
            else:
                assert isinstance(node.first, SelectionTree.Node)
                self._SynthesizeNode(node.first, insn16, insn32)

        with _else():

            if isinstance(node.second, CommandDesc):
                assert node.second.mapTo is not None
                # comment(f"{node.second} -> {node.second.mapTo.targetCmd}")
                t = CommandTransform(node.second)
                expr = t.SynthesizeExpression(insn16)
                if insn32.isReg:
                    insn32 <<= expr
                else:
                    insn32 //= expr
            else:
                assert isinstance(node.second, SelectionTree.Node)
                self._SynthesizeNode(node.second, insn16, insn32)


def Synthesize(input: InputNet[Wire, 16], output: OutputNet[Reg, 30]):
    """Synthesize decompressor logic. It results into a bunch of _if/_else statements and
    continuous assignments to the output. So it should called in context of combinational procedural
    block.

    :param input: 16 bits net with compressed instruction.
    :param output: 30 bits net with decompressed instruction. Two LSB always are 'b11 for 32 bits
        instruction, so do not use them.
    """
    selTree = SelectionTree.Generate(commands16.values())
    selTree.Synthesize(input, output)


DefineCommands32()
DefineCommands16()
