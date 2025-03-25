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
                allocator.destroy(ptr);
            }
        },
        .reversed => {
            while (list.pop()) |ptr| {
                try testing.expect(std.mem.allEqual(u8, std.mem.asBytes(ptr), 0x41));
                allocator.destroy(ptr);
            }
        },
        .random => {
            var ascon = std.Random.Ascon.init([_]u8{0x42} ** 32);
            const rand = ascon.random();

            while (list.getLastOrNull()) |_| {
                const ptr = list.swapRemove(rand.intRangeAtMost(usize, 0, list.items.len - 1));
                try testing.expect(std.mem.allEqual(u8, std.mem.asBytes(ptr), 0x41));
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

fn umm_test_random_size(comptime free_order: FreeOrder) !void {
    const Umm = UmmAllocator(.{});
    var buf: [std.math.maxInt(u15) * 8]u8 align(16) = undefined;
    var umm = try Umm.init(&buf);
    defer std.testing.expect(umm.deinit() == .ok) catch @panic("leak");
    const allocator = umm.allocator();

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
                allocator.free(ptr);
            }
        },
        .reversed => {
            while (list.pop()) |ptr| {
                try testing.expect(std.mem.allEqual(u8, ptr, 0x41));
                allocator.free(ptr);
            }
        },
        .random => {
            while (list.getLastOrNull()) |_| {
                const ptr = list.swapRemove(rand.intRangeAtMost(usize, 0, list.items.len - 1));
                try testing.expect(std.mem.allEqual(u8, ptr, 0x41));
                allocator.free(ptr);
            }
        },
    }
}

test "random size allocations - free in same order" {
    try umm_test_random_size(.normal);
}
test "random size allocations - free in reverse order" {
    try umm_test_random_size(.reversed);
}
test "random size allocations - free in random order" {
    try umm_test_random_size(.random);
}

// fn umm_test_random_size_and_random_alignment(comptime free_order: FreeOrder) !void {
//     const Umm = UmmAllocator(.{});
//     var buf: [std.math.maxInt(u15) * 8]u8 align(16) = undefined;
//     var umm = try Umm.init(&buf);
//     defer std.testing.expect(umm.deinit() == .ok) catch @panic("leak");
//     const allocator = umm.allocator();

//     var list = std.ArrayList([]u8).init(if (@import("builtin").is_test) testing.allocator else std.heap.page_allocator);
//     defer list.deinit();

//     var ascon = std.rand.Ascon.init([_]u8{0x42} ** 32);
//     const rand = ascon.random();

//     var i: usize = 0;
//     while (i < 256) : (i += 1) {
//         const size = rand.intRangeLessThanBiased(usize, 0, 1024);
//         const alignment = std.math.pow(u20, 2, rand.intRangeAtMostBiased(u20, 0, 5));
//         const ptr = try allocator.allocWithOptions(u8, size, alignment, null);
//         @memset(ptr, 0x41);
//         try list.append(ptr);
//     }

//     switch (free_order) {
//         .normal => {
//             for (list.items) |ptr| {
//                 try testing.expect(std.mem.allEqual(u8, ptr, 0x41));
//                 allocator.free(ptr);
//             }
//         },
//         .reversed => {
//             while (list.popOrNull()) |ptr| {
//                 try testing.expect(std.mem.allEqual(u8, ptr, 0x41));
//                 allocator.free(ptr);
//             }
//         },
//         .random => {
//             while (list.getLastOrNull()) |_| {
//                 const ptr = list.swapRemove(rand.intRangeAtMost(usize, 0, list.items.len - 1));
//                 try testing.expect(std.mem.allEqual(u8, ptr, 0x41));
//                 allocator.free(ptr);
//             }
//         },
//     }
// }
// test "random size allocations with random alignment - free in same order" {
//     try umm_test_random_size_and_random_alignment(.normal);
// }
// test "random size allocations with random alignment - free in reverse order" {
//     try umm_test_random_size_and_random_alignment(.reversed);
// }
// test "random size allocations with random alignment - free in random order" {
//     try umm_test_random_size_and_random_alignment(.random);
// }
