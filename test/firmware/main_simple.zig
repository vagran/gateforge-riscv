// These symbols come from the linker script
extern var _data: u32;
extern const _edata: u32;
extern var _bss: u32;
extern const _ebss: u32;
extern const _sstack: u32;

const outDataPtr: *volatile u32 = @ptrFromInt(0x800);


export fn _resetHandler() linksection(".reset") noreturn {
    _start();
}

export fn _start() noreturn {
    // Clear the BSS
    const bss: [*]u8 = @ptrCast(&_bss);
    const bss_size = @intFromPtr(&_ebss) - @intFromPtr(&_bss);
    for (bss[0..bss_size]) |*b| b.* = 0;

    asm volatile ("la sp, _sstack");    // set stack pointer

    main();

    asm volatile ("ebreak");
    while (true) {}
}

fn main() void {
    outDataPtr.* = 0xdeadbeef;
}
