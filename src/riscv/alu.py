from typing import List
from gateforge.core import Net, Wire
from gateforge.dsl import namespace, wire


class Alu:
    size: int
    inA: Net
    inB: Net
    # True to perform subtraction, addition otherwise
    isSub: Net

    outOr: Wire
    outAnd: Wire
    outXor: Wire
    # Addition or subtraction result, depending on `isSub` input
    outAddSub: Wire
    # ADD/SUB trimmed result is zero (including carry)
    outZ: Wire
    # Signed subtraction result is negative
    outLt: Wire
    # Unsigned subtraction result is negative
    outLtu: Wire

    # Internal carry bits, do not use bits vector since Verilator thinks it has cyclic dependency.
    _carry: List[Wire]


    def __init__(self, inA: Net, inB: Net, isSub: Net, *, size: int = 32):
        self.size = size
        self.inA = inA
        self.inB = inB
        self.isSub = isSub

        with namespace("RiscvAlu"):
            self.outOr = wire("outOr", size)
            self.outAnd = wire("outAnd", size)
            self.outXor = wire("outXor", size)
            self.outAddSub = wire("outAddSub", size)
            self.outZ = wire("outZ")
            self.outLt = wire("outLt")
            self.outLtu = wire("outLtu")

            self._carry = [wire(f"_carry{i}") for i in range(size)]


    def __call__(self):
        self.outOr <<= self.inA | self.inB
        self.outAnd <<= self.inA & self.inB
        self.outXor <<= self.inA ^ self.inB

        # Subtraction is implemented as addition with twos complement of B which is calculated as
        # inversion of all bits of B (by xor'ing with `isSub`) and adding one (specifying input
        # carry bit).
        for i in range(self.size):
            cIn = self._carry[i - 1] if i > 0 else self.isSub
            b = self.inB[i] ^ self.isSub
            self.outAddSub[i] <<= self.inA[i] ^ b ^ cIn
            self._carry[i] <<= (self.inA[i] & b) | ((self.inA[i] | b) & cIn)

        # All result bits are  zero
        self.outZ <<= self.outAddSub.reduce_nor
        # Subtraction carry is inverse of twos complement addition carry
        self.outLtu <<= ~self._carry[self.size - 1]
        # LT is inverted LTU if operands have different signs
        self.outLt <<= self.outLtu ^ self.outXor[self.size - 1]
