# RISC-V example implementation using GateForge framework

This projects demonstrates how to use [GateForge](https://github.com/vagran/GateForge) framework to
implement RISC-V processor. For now, it is just a learning sample, and is not intended for use in
production projects. However, it may change in future, once I will battle test it in my production.

It implements base RV32I profile and optional "E" (embedded) and "C" (compressed instruction set)
profiles, as an example of design parameterization.

See unit tests for an example of instantiation and wiring. It has memory and control interfaces
which should be connected.

## Internals

`instruction_set.py` file decomposes RV32IC ISA.
 - Implements generation of instruction decompressing unit, so that it is not relied on manually
   crafted error-prone decompression logic.
 - Provides op-codes generation (basically assembler) for use, for example, in unit tests.

## Tests

There are several levels of tests in this project. All tests are based on Python `unittest` built-in
package.
 - Unit tests for separate design components, like ALU or instruction decompressor.
 - Running emulated CPU to tests separate instructions or code fragments. Simulation relies on
   Verilator integration provided by GateForge framework.
 - Compiling more complex firmware and running it on the emulated CPU. Zig language was used for the
   test firmware.

### Compiling test firmware

Use the following command to build test firmware in `test/firmware` directory:
```bash
zig build -Dname=simple -Dcompressed=false
```
`name` parameter specifies firmware variant name. Currently there are `simple`, `print` and `spigot`
values accepted. `spigot` firmware is the most complex one and serves as end-to-end ultimate test.
It uses custom memory allocator, custom arbitrary precision integer arithmetic implementation and
Spigot algorithm to calculate some number (defined in the test) of Pi digits, which are then
verified with the expected sequence. `compressed` parameter selects if compressed instructions (`C`
extension is enabled for target).

Run `build.sh path_to_zig_compiler` to build all firmware variants.

TODO Check and report FPGA utilization on a sample board.
