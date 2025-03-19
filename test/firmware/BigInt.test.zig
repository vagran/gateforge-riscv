const std = @import("std");

const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

const bi = @import("BigInt.zig");
const BigInt = bi.BigInt;
const Sign = bi.Sign;
const mul32 = bi.mul32;
const div64by32 = bi.div64by32;


fn expectEqualSlices(comptime T: type, expected: []const T, actual: []const T) !void {
    try expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try expectEqual(e, a);
    }
}

test "mul32: basic multiplication" {
    const res = mul32(2, 3);
    try expectEqual(@as(u32, 0), res.hi);
    try expectEqual(@as(u32, 6), res.lo);
}

test "mul32: max values multiplication" {
    const res = mul32(0xFFFFFFFF, 0xFFFFFFFF);
    try expectEqual(@as(u32, 0xFFFFFFFE), res.hi);
    try expectEqual(@as(u32, 0x00000001), res.lo);
}

test "mul32: multiply by zero" {
    const res = mul32(12345, 0);
    try expectEqual(@as(u32, 0), res.hi);
    try expectEqual(@as(u32, 0), res.lo);
}

test "mul32: zero multiplied by zero" {
    const res = mul32(0, 0);
    try expectEqual(@as(u32, 0), res.hi);
    try expectEqual(@as(u32, 0), res.lo);
}

test "mul32: multiply by one" {
    const res = mul32(0xFFFFFFFF, 1);
    try expectEqual(@as(u32, 0), res.hi);
    try expectEqual(@as(u32, 0xFFFFFFFF), res.lo);
}

test "mul32: power of two multiplication" {
    const res = mul32(0x10000, 0x10000);
    try expectEqual(@as(u32, 1), res.hi);
    try expectEqual(@as(u32, 0), res.lo);
}

test "mul32: large and small numbers" {
    const res = mul32(0x80000000, 2);
    try expectEqual(@as(u32, 1), res.hi);
    try expectEqual(@as(u32, 0), res.lo);
}

test "mul32: carry handling" {
    const res = mul32(0xFFFFFFFF, 2);
    try expectEqual(@as(u32, 1), res.hi);
    try expectEqual(@as(u32, 0xFFFFFFFE), res.lo);
}

test "mul32: commutative property" {
    const res1 = mul32(123, 456);
    const res2 = mul32(456, 123);
    try expectEqual(res1.hi, res2.hi);
    try expectEqual(res1.lo, res2.lo);
}

test "mul32: intermediate overflow" {
    const res = mul32(0xDEADBEEF, 0xCAFEBABE);
    // Pre-calculated expected result
    try expectEqual(@as(u32, 0xb092ab7b), res.hi);
    try expectEqual(@as(u32, 0x88cf5b62), res.lo);
}

test "add: positive + positive (no carry)" {
    const a = try BigInt.init(std.testing.allocator, &.{3}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{4}, .positive);
    defer b.deinit();

    const res = try BigInt.add(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{7}, res.limbs);
}

test "add: positive + positive (with carry)" {
    const a = try BigInt.init(std.testing.allocator, &.{0xFFFFFFFF}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{1}, .positive);
    defer b.deinit();

    const res = try BigInt.add(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{0, 1}, res.limbs);
}

test "add: negative + negative" {
    const a = try BigInt.init(std.testing.allocator, &.{5}, .negative);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{3}, .negative);
    defer b.deinit();

    const res = try BigInt.add(a, b);
    defer res.deinit();

    try expectEqual(Sign.negative, res.sign);
    try expectEqualSlices(u32, &.{8}, res.limbs);
}

test "add: positive + negative (positive result)" {
    const a = try BigInt.init(std.testing.allocator, &.{5}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{3}, .negative);
    defer b.deinit();

    const res = try BigInt.add(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{2}, res.limbs);
}

test "add: positive + negative (negative result)" {
    const a = try BigInt.init(std.testing.allocator, &.{3}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{5}, .negative);
    defer b.deinit();

    const res = try BigInt.add(a, b);
    defer res.deinit();

    try expectEqual(Sign.negative, res.sign);
    try expectEqualSlices(u32, &.{2}, res.limbs);
}

test "add: zero + value" {
    const a = try BigInt.init(std.testing.allocator, &.{0}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{0xFFFFFFFF}, .positive);
    defer b.deinit();

    const res = try BigInt.add(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{0xFFFFFFFF}, res.limbs);
}

test "add: cross-limb carry propagation" {
    const a = try BigInt.init(std.testing.allocator, &.{0xFFFFFFFF, 0xFFFFFFFF}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{1}, .positive);
    defer b.deinit();

    const res = try BigInt.add(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{0, 0, 1}, res.limbs);
}

test "add: different limb counts" {
    const a = try BigInt.init(std.testing.allocator, &.{1, 2}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{3}, .positive);
    defer b.deinit();

    const res = try BigInt.add(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{4, 2}, res.limbs);
}

test "add: result becomes zero" {
    const a = try BigInt.init(std.testing.allocator, &.{5}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{5}, .negative);
    defer b.deinit();

    const res = try BigInt.add(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{0}, res.limbs);
}

test "add: very large numbers" {
    const a = try BigInt.init(std.testing.allocator, &.{
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    }, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    }, .positive);
    defer b.deinit();

    const res = try BigInt.add(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{
        0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 1
    }, res.limbs);
}

test "sub: positive - positive (no borrow)" {
    const a = try BigInt.init(std.testing.allocator, &.{5}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{3}, .positive);
    defer b.deinit();

    const res = try BigInt.sub(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{2}, res.limbs);
}

test "sub: positive - positive (negative result)" {
    const a = try BigInt.init(std.testing.allocator, &.{3}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{5}, .positive);
    defer b.deinit();

    const res = try BigInt.sub(a, b);
    defer res.deinit();

    try expectEqual(Sign.negative, res.sign);
    try expectEqualSlices(u32, &.{2}, res.limbs);
}

test "sub: negative - negative" {
    const a = try BigInt.init(std.testing.allocator, &.{5}, .negative);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{3}, .negative);
    defer b.deinit();

    const res = try BigInt.sub(a, b);
    defer res.deinit();

    try expectEqual(Sign.negative, res.sign);
    try expectEqualSlices(u32, &.{2}, res.limbs);
}

test "sub: positive - negative" {
    const a = try BigInt.init(std.testing.allocator, &.{5}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{3}, .negative);
    defer b.deinit();

    const res = try BigInt.sub(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{8}, res.limbs);
}

test "sub: negative - positive" {
    const a = try BigInt.init(std.testing.allocator, &.{5}, .negative);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{3}, .positive);
    defer b.deinit();

    const res = try BigInt.sub(a, b);
    defer res.deinit();

    try expectEqual(Sign.negative, res.sign);
    try expectEqualSlices(u32, &.{8}, res.limbs);
}

test "sub: subtract zero" {
    const a = try BigInt.init(std.testing.allocator, &.{5}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{0}, .positive);
    defer b.deinit();

    const res = try BigInt.sub(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{5}, res.limbs);
}

test "sub: zero - positive" {
    const a = try BigInt.init(std.testing.allocator, &.{0}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{5}, .positive);
    defer b.deinit();

    const res = try BigInt.sub(a, b);
    defer res.deinit();

    try expectEqual(Sign.negative, res.sign);
    try expectEqualSlices(u32, &.{5}, res.limbs);
}

test "sub: cross-limb borrow" {
    const a = try BigInt.init(std.testing.allocator, &.{0, 1}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{1}, .positive);
    defer b.deinit();

    const res = try BigInt.sub(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{0xFFFFFFFF}, res.limbs); // Changed expected value
}

test "sub: different limb counts" {
    const a = try BigInt.init(std.testing.allocator, &.{3, 2}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{1}, .positive);
    defer b.deinit();

    const res = try BigInt.sub(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{2, 2}, res.limbs);
}

test "sub: result zero" {
    const a = try BigInt.init(std.testing.allocator, &.{5}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{5}, .positive);
    defer b.deinit();

    const res = try BigInt.sub(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{0}, res.limbs);
}

test "sub: large numbers" {
    const a = try BigInt.init(std.testing.allocator, &.{
        0x00000000, 0x00000001  // Represents 0x100000000
    }, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{
        0x00000001  // Represents 0x1
    }, .positive);
    defer b.deinit();

    const res = try BigInt.sub(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{0xFFFFFFFF}, res.limbs); // Now expects 1 limb
}

test "mul: basic positive multiplication" {
    const a = try BigInt.init(std.testing.allocator, &.{3}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{4}, .positive);
    defer b.deinit();

    const res = try BigInt.mul(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{12}, res.limbs);
}

test "mul: positive * negative" {
    const a = try BigInt.init(std.testing.allocator, &.{5}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{3}, .negative);
    defer b.deinit();

    const res = try BigInt.mul(a, b);
    defer res.deinit();

    try expectEqual(Sign.negative, res.sign);
    try expectEqualSlices(u32, &.{15}, res.limbs);
}

test "mul: negative * negative" {
    const a = try BigInt.init(std.testing.allocator, &.{5}, .negative);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{3}, .negative);
    defer b.deinit();

    const res = try BigInt.mul(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{15}, res.limbs);
}

test "mul: multiply by zero" {
    const a = try BigInt.init(std.testing.allocator, &.{0xFFFFFFFF}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{0}, .positive);
    defer b.deinit();

    const res = try BigInt.mul(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{0}, res.limbs);
}

test "mul: zero * negative" {
    const a = try BigInt.init(std.testing.allocator, &.{0}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{5}, .negative);
    defer b.deinit();

    const res = try BigInt.mul(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{0}, res.limbs);
}

test "mul: max value multiplication" {
    const a = try BigInt.init(std.testing.allocator, &.{0xFFFFFFFF}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{0xFFFFFFFF}, .positive);
    defer b.deinit();

    const res = try BigInt.mul(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{0x00000001, 0xFFFFFFFE}, res.limbs);
}

test "mul: cross-limb carry" {
    const a = try BigInt.init(std.testing.allocator, &.{0xFFFFFFFF, 0x1}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{0x2}, .positive);
    defer b.deinit();

    const res = try BigInt.mul(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{0xFFFFFFFE, 0x3}, res.limbs);
}

test "mul: power of two multiplication" {
    const a = try BigInt.init(std.testing.allocator, &.{0x80000000}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{0x2}, .positive);
    defer b.deinit();

    const res = try BigInt.mul(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{0x00000000, 0x1}, res.limbs);
}

test "mul: identity multiplication" {
    const a = try BigInt.init(std.testing.allocator, &.{0x12345678}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{1}, .positive);
    defer b.deinit();

    const res = try BigInt.mul(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{0x12345678}, res.limbs);
}

test "mul: commutative property" {
    const a = try BigInt.init(std.testing.allocator, &.{0x1234, 0x5678}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{0x9ABC}, .negative);
    defer b.deinit();

    const res1 = try BigInt.mul(a, b);
    defer res1.deinit();
    const res2 = try BigInt.mul(b, a);
    defer res2.deinit();

    try expectEqual(res1.sign, res2.sign);
    try expectEqualSlices(u32, res1.limbs, res2.limbs);
}

test "mul: normalization check" {
    const a = try BigInt.init(std.testing.allocator, &.{0, 5}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{0, 3}, .positive);
    defer b.deinit();

    const res = try BigInt.mul(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{0, 0, 15}, res.limbs);
}

test "div: basic division" {
    const a = try BigInt.init(std.testing.allocator, &.{15}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{3}, .positive);
    defer b.deinit();

    const res = try BigInt.div(a, b);
    defer res.deinit();
    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{5}, res.limbs);
}

test "div: negative dividend" {
    const a = try BigInt.init(std.testing.allocator, &.{15}, .negative);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{3}, .positive);
    defer b.deinit();

    const res = try BigInt.div(a, b);
    defer res.deinit();
    try expectEqual(Sign.negative, res.sign);
    try expectEqualSlices(u32, &.{5}, res.limbs);
}

test "div: single limb division" {
    const a = try BigInt.init(std.testing.allocator, &.{0xFFFFFFFE, 0x1}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{2}, .positive);
    defer b.deinit();

    const res = try BigInt.div(a, b);
    defer res.deinit();

    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{0xFFFFFFFF}, res.limbs);
}

test "div: multi-limb division" {
    const a = try BigInt.init(std.testing.allocator, &.{0xFFFFFFFE, 0x1}, .positive); // 0x1FFFFFFFE
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{0x2}, .positive);
    defer b.deinit();

    const res = try BigInt.div(a, b);
    defer res.deinit();
    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{0xFFFFFFFF}, res.limbs);
}

test "mod: basic modulus" {
    const a = try BigInt.init(std.testing.allocator, &.{15}, .positive);
    defer a.deinit();
    const b = try BigInt.init(std.testing.allocator, &.{7}, .positive);
    defer b.deinit();

    const res = try BigInt.mod(a, b);
    defer res.deinit();
    try expectEqual(Sign.positive, res.sign);
    try expectEqualSlices(u32, &.{1}, res.limbs);
}

test "div64by32: basic division" {
    const res = div64by32(0, 6, 2);
    try expectEqual(3, res.q);
    try expectEqual(0, res.r);
}

test "div64by32: division with remainder" {
    const res = div64by32(0, 7, 2);
    try expectEqual(3, res.q);
    try expectEqual(1, res.r);
}

test "div64by32: high bit set" {
    const res = div64by32(1, 0, 2);
    try expect(res.q == 0x80000000);
    try expect(res.r == 0);
}

test "div64by32: large values" {
    const res = div64by32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    try expect(res.q == 1);
    try expect(res.r == 0);
}

test "div64by32: zero dividend" {
    const res = div64by32(0, 0, 5);
    try expect(res.q == 0);
    try expect(res.r == 0);
}

test "div64by32: divisor larger than dividend" {
    const res = div64by32(0, 3, 5);
    try expect(res.q == 0);
    try expect(res.r == 3);
}

test "div64by32: full 32-bit quotient" {
    const res = div64by32(0, 0xFFFFFFFF, 1);
    try expect(res.q == 0xFFFFFFFF);
    try expect(res.r == 0);
}

test "div64by32: carry between halves" {
    const res = div64by32(1, 0, 3);
    try expect(res.q == 0x55555555);
    try expect(res.r == 1);
}

test "div64by32: complex division" {
    // 0x00000000FFFFFFFF / 0x10000 = 0x0000FFFF remainder 0xFFFF
    const res = div64by32(0x00000000, 0xFFFFFFFF, 0x10000);
    try expect(res.q == 0x0000FFFF);
    try expect(res.r == 0xFFFF);
}
