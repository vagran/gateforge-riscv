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

    main() catch unreachable;

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

var heap: [0x1000]u8 = undefined;

const UmmAllocator = @import("umm.zig").UmmAllocator;

fn main() !void {
    const testValue = hostIo.inData;

    const Allocator = UmmAllocator(.{});
    var allocInst = try Allocator.init(&heap);
    defer _ = allocInst.deinit();
    const allocator = allocInst.allocator();
    const s = try std.fmt.allocPrint(allocator, "Test value: {x}", .{testValue});
    defer allocator.free(s);
    @memcpy(&hostIo.outString, s);
}
