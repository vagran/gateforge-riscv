import io
import os


disableVerilatorTests = int(os.getenv("DISABLE_VERILATOR_TESTS", "0"))


class NullOutput(io.StringIO):
    def write(self, _, /):
        pass
