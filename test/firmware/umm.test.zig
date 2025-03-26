// https://github.com/ZigEmbeddedGroup/umm-zig

const std = @import("std");
const testing = std.testing;

const UmmAllocator = @import("umm.zig").UmmAllocator;

const FreeOrder = enum {
    normal,
    reversed,
    random,
};

fn umm_test_type(comptime T: type, comptime free_order: FreeOrder) !void {
    const Umm = UmmAllocator(.{});
    var buf: [std.math.maxInt(u15) * 8]u8 align(16) = undefined;
    var umm = try Umm.init(&buf);
    defer std.testing.expect(umm.deinit() == .ok) catch @panic("leak");
    const allocator = umm.allocator();

    var list = std.ArrayList(*T).init(if (@import("builtin").is_test) testing.allocator else std.heap.page_allocator);
    defer list.deinit();

    var i: usize = 0;
    while (i < 1024) : (i += 1) {
        const ptr = try allocator.create(T);
        @memset(std.mem.asBytes(ptr), 0x41);
        try list.append(ptr);
    }

    switch (free_order) {
        .normal => {
            for (list.items) |ptr| {
                try testing.expect(std.mem.allEqual(u8, std.mem.asBytes(ptr), 0x41));
                @memset(std.mem.asBytes(ptr), 0xfe);
                allocator.destroy(ptr);
            }
        },
        .reversed => {
            while (list.pop()) |ptr| {
                try testing.expect(std.mem.allEqual(u8, std.mem.asBytes(ptr), 0x41));
                @memset(std.mem.asBytes(ptr), 0xfe);
                allocator.destroy(ptr);
            }
        },
        .random => {
            var rng = std.Random.DefaultPrng.init(0);
            const rand = rng.random();

            while (list.getLastOrNull()) |_| {
                const ptr = list.swapRemove(rand.intRangeAtMost(usize, 0, list.items.len - 1));
                try testing.expect(std.mem.allEqual(u8, std.mem.asBytes(ptr), 0x41));
                @memset(std.mem.asBytes(ptr), 0xfe);
                allocator.destroy(ptr);
            }
        },
    }
}

test "u16 allocations - free in same order" {
    try umm_test_type(u16, .normal);
}
test "u16 allocations - free in reverse order" {
    try umm_test_type(u16, .reversed);
}
test "u16 allocations - free in random order" {
    try umm_test_type(u16, .random);
}

test "u32 allocations - free in same order" {
    try umm_test_type(u32, .normal);
}
test "u32 allocations - free in reverse order" {
    try umm_test_type(u32, .reversed);
}
test "u32 allocations - free in random order" {
    try umm_test_type(u32, .random);
}

test "u64 allocations - free in same order" {
    try umm_test_type(u64, .normal);
}
test "u64 allocations - free in reverse order" {
    try umm_test_type(u64, .reversed);
}
test "u64 allocations - free in random order" {
    try umm_test_type(u64, .random);
}

const Foo = struct {
    a: []u8,
    b: u32,
    c: u64,
};
test "Foo allocations - free in same order" {
    try umm_test_type(Foo, .normal);
}
test "Foo allocations - free in reverse order" {
    try umm_test_type(Foo, .reversed);
}
test "Foo allocations - free in random order" {
    try umm_test_type(Foo, .random);
}

fn umm_test_random_size(comptime free_order: FreeOrder, numPasses: u32) !void {
    const Umm = UmmAllocator(.{});
    var buf: [std.math.maxInt(u15) * 8]u8 align(16) = undefined;
    var umm = try Umm.init(&buf);
    defer std.testing.expect(umm.deinit() == .ok) catch @panic("leak");
    const allocator = umm.allocator();

    for (0..numPasses) |_| {
        var list = std.ArrayList([]u8).init(if (@import("builtin").is_test) testing.allocator else std.heap.page_allocator);
        defer list.deinit();

        var ascon = std.Random.Ascon.init([_]u8{0x42} ** 32);
        const rand = ascon.random();

        var i: usize = 0;
        while (i < 256) : (i += 1) {
            const size = rand.intRangeLessThanBiased(usize, 0, 1024);
            const ptr = try allocator.alloc(u8, size);
            @memset(ptr, 0x41);
            try list.append(ptr);
        }

        switch (free_order) {
            .normal => {
                for (list.items) |ptr| {
                    try testing.expect(std.mem.allEqual(u8, ptr, 0x41));
                    @memset(ptr, 0xfe);
                    allocator.free(ptr);
                }
            },
            .reversed => {
                while (list.pop()) |ptr| {
                    try testing.expect(std.mem.allEqual(u8, ptr, 0x41));
                    @memset(ptr, 0xfe);
                    allocator.free(ptr);
                }
            },
            .random => {
                while (list.getLastOrNull()) |_| {
                    const ptr = list.swapRemove(rand.intRangeAtMost(usize, 0, list.items.len - 1));
                    try testing.expect(std.mem.allEqual(u8, ptr, 0x41));
                    @memset(ptr, 0xfe);
                    allocator.free(ptr);
                }
            },
        }
    }
}

test "random size allocations - free in same order" {
    try umm_test_random_size(.normal, 1);
}
test "random size allocations - free in reverse order" {
    try umm_test_random_size(.reversed, 1);
}
test "random size allocations - free in random order" {
    try umm_test_random_size(.random, 1);
}

test "random size allocations - free in random order - multiple passes" {
    try umm_test_random_size(.random, 10);
}

test "random allocations and frees within memory limit" {
    const Umm = UmmAllocator(.{});
    var buf: [std.math.maxInt(u15) * 8]u8 align(16) = undefined;
    var umm = try Umm.init(&buf);
    defer std.testing.expect(umm.deinit() == .ok) catch @panic("leak");
    const allocator = umm.allocator();

    // Initialize PRNG with deterministic seed
    var rng = std.Random.DefaultPrng.init(0);
    const rand = rng.random();

    // List to track allocated blocks
    var allocated = std.ArrayList([]u8).init(testing.allocator);
    defer allocated.deinit();

    const max_block_size = 32; // Maximum size for individual allocations
    const iterations = 10_000; // Number of operations to perform

    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        // Randomly choose between allocation and free
        if (rand.boolean()) {
            // Attempt allocation
            const size = rand.intRangeAtMost(usize, 4, max_block_size);
            const result = allocator.alloc(u8, size);

            if (result) |block| {
                // Verify and use the memory
                @memset(block, 0x41);
                allocated.append(block) catch unreachable;
            } else |err| switch (err) {
                error.OutOfMemory => {
                    // Handle OOM by freeing random existing block
                    if (allocated.items.len > 0) {
                        const idx = rand.intRangeLessThan(usize, 0, allocated.items.len);
                        allocator.free(allocated.swapRemove(idx));
                    }
                },
                else => unreachable,
            }
        } else {
            // Free random block if available
            if (allocated.items.len > 0) {
                const idx = rand.intRangeLessThan(usize, 0, allocated.items.len);
                const block = allocated.swapRemove(idx);
                @memset(block, 0xfe);
                allocator.free(block);
            }
        }
    }

    // Cleanup remaining allocations
    while (allocated.pop()) |block| {
        try testing.expect(std.mem.allEqual(u8, block, 0x41));
        @memset(block, 0xfe);
        allocator.free(block);
    }
}
