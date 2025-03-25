// These symbols come from the linker script
extern var _data: u32;
extern const _edata: u32;
extern var _bss: u32;
extern const _ebss: u32;
extern const _sstack: u32;

const std = @import("std");

const HostIo = extern struct {
    panic: [4]u8,
    outData: u32,
    inData: u64,
    outString: [128]u8
};

const hostIo: *volatile HostIo = @ptrFromInt(0x800);

export fn _resetHandler() linksection(".reset") noreturn {
    // Set stack pointer
    asm volatile ("la sp, _sstack");
    _start();
}

export fn _start() noreturn {
    // Clear the BSS
    const bss: [*]u8 = @ptrCast(&_bss);
    const bss_size = @intFromPtr(&_ebss) - @intFromPtr(&_bss);
    for (bss[0..bss_size]) |*b| b.* = 0;

    @memset(&hostIo.panic, 0);

    main();

    asm volatile ("ebreak");
    while (true) {}
}

pub const panic = std.debug.FullPanic(_panic);

fn _panic(msg: []const u8, first_trace_addr: ?usize) noreturn {
    hostIo.panic = "PNIC";
    _ = msg;
    _ = first_trace_addr;
    asm volatile ("ebreak");
    while (true) {}
}

export fn TestCopy(dst: [*]u8, src: [*]const u8, size: usize) void {
    @memcpy(dst[0..size], src);
}

export fn Hash(x1: u32, x2: u32, seed: u32) u32 {
    var h: u32 = seed;
    const c1: u32 = 0xcc9e2d51;
    const c2: u32 = 0x1b873593;

    const input: [2]u32 = .{x1, x2};

    for (input) |x| {
        var k: u32 = x;
        k *%= c1;
        k = (k << 15) | (k >> (32 - 15));  // Rotate left 15 bits
        k *%= c2;

        h ^= k;
        h = (h << 13) | (h >> (32 - 13));  // Rotate left 13 bits
        h = h *% 5 +% 0xe6546b64;
    }

    // Return as unsigned 32-bit integer
    return h;
}

fn main() void {
    hostIo.outData = Hash(@truncate(hostIo.inData), @truncate(hostIo.inData >> 32), 0xc001babe);
    TestCopy(@volatileCast(&hostIo.outString), "1234567890", 10);
}
