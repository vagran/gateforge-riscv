import io
import os
from pathlib import Path

from gateforge.verilator import VerilatorParams


disableVerilatorTests = int(os.getenv("DISABLE_VERILATOR_TESTS", "0"))

workspaceDir = Path(__file__).parent / "workspace"


class NullOutput(io.StringIO):
    def write(self, _, /):
        pass


def GetVerilatorParams(testName: str) -> VerilatorParams:
    return VerilatorParams(buildDir=str(workspaceDir / testName), quite=False)


def SignExtend(value, size=32):
    signBit = 1 << (size - 1)
    return (value & (signBit - 1)) - (value & signBit)
