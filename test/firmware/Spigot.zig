const std = @import("std");
const BigInt = @import("BigInt.zig").BigInt;
const Sign = @import("BigInt.zig").Sign;


fn spigot_pi(buffer: [] u8, allocator: std.mem.Allocator) !void {
    // Initialize constants
    const zero = try BigInt.init(allocator, &.{0}, .positive);
    defer zero.deinit();
    const one = try BigInt.init(allocator, &.{1}, .positive);
    defer one.deinit();
    const two = try BigInt.init(allocator, &.{2}, .positive);
    defer two.deinit();
    const three = try BigInt.init(allocator, &.{3}, .positive);
    defer three.deinit();
    const four = try BigInt.init(allocator, &.{4}, .positive);
    defer four.deinit();
    const seven = try BigInt.init(allocator, &.{7}, .positive);
    defer seven.deinit();
    const ten = try BigInt.init(allocator, &.{10}, .positive);
    defer ten.deinit();

    // Initialize variables
    var q = try one.copy();
    defer q.deinit();
    var r = try zero.copy();
    defer r.deinit();
    var t = try one.copy();
    defer t.deinit();
    var k = try one.copy();
    defer k.deinit();
    var n = try three.copy();
    defer n.deinit();
    var l = try three.copy();
    defer l.deinit();

    var i: usize = 0;
    while (i < buffer.len) {
        // Calculate 4*q + r - t
        const four_q = try q.mul(four);
        defer four_q.deinit();
        const four_q_plus_r = try four_q.add(r);
        defer four_q_plus_r.deinit();
        const numerator = try four_q_plus_r.sub(t);
        defer numerator.deinit();

        // Calculate n*t
        const n_t = try n.mul(t);
        defer n_t.deinit();

        if (numerator.compare(n_t) == .lt) {
            // Store digit
            buffer[i] = try bigIntToDigit(n);
            i += 1;

            // Calculate new values
            const nr1 = try n.mul(t);
            defer nr1.deinit();
            const r_sub = try r.sub(nr1);
            defer r_sub.deinit();
            const nr = try r_sub.mul(ten);
            defer nr.deinit();

            const three_q = try q.mul(three);
            defer three_q.deinit();
            const three_q_plus_r = try three_q.add(r);
            defer three_q_plus_r.deinit();
            const numerator2 = try three_q_plus_r.mul(ten);
            defer numerator2.deinit();
            const n_new = try numerator2.div(t);
            defer n_new.deinit();
            const n_10 = try ten.mul(n);
            defer n_10.deinit();
            const n_new_sub = try n_new.sub(n_10);
            defer n_new_sub.deinit();

            const q_new = try q.mul(ten);
            q.deinit();
            q = q_new;

            const r_new = try nr.copy();
            r.deinit();
            r = r_new;

            const n_new_final = try n_new_sub.copy();
            n.deinit();
            n = n_new_final;

        } else {
            // Update values
            const two_q = try q.mul(two);
            defer two_q.deinit();
            const two_q_plus_r = try two_q.add(r);
            defer two_q_plus_r.deinit();
            const nr = try two_q_plus_r.mul(l);
            defer nr.deinit();

            const seven_k = try k.mul(seven);
            defer seven_k.deinit();
            const seven_k_plus_two = try seven_k.add(two);
            defer seven_k_plus_two.deinit();
            const q_seven = try q.mul(seven_k_plus_two);
            defer q_seven.deinit();
            const r_l = try r.mul(l);
            defer r_l.deinit();
            const numerator2 = try q_seven.add(r_l);
            defer numerator2.deinit();
            const denominator = try t.mul(l);
            defer denominator.deinit();
            const nn = try numerator2.div(denominator);
            defer nn.deinit();

            const q_new = try q.mul(k);
            q.deinit();
            q = q_new;

            const t_new = try t.mul(l);
            t.deinit();
            t = t_new;

            const l_new = try l.add(two);
            l.deinit();
            l = l_new;

            const k_new = try k.add(one);
            k.deinit();
            k = k_new;

            const n_new = try nn.copy();
            n.deinit();
            n = n_new;

            const r_new = try nr.copy();
            r.deinit();
            r = r_new;
        }
    }
}

fn bigIntToDigit(num: BigInt) !u8 {
    if (num.limbs.len != 1) return error.InvalidDigit;
    const value = num.limbs[0];
    if (value > 9) return error.InvalidDigit;
    return @intCast(value);
}


const Pi =
\\ 31415926535 8979323846 2643383279 5028841971 6939937510
\\  5820974944 5923078164 0628620899 8628034825 3421170679
\\  8214808651 3282306647 0938446095 5058223172 5359408128
\\  4811174502 8410270193 8521105559 6446229489 5493038196
\\  4428810975 6659334461 2847564823 3786783165 2712019091
\\  4564856692 3460348610 4543266482 1339360726 0249141273
\\  7245870066 0631558817 4881520920 9628292540 9171536436
\\  7892590360 0113305305 4882046652 1384146951 9415116094
\\  3305727036 5759591953 0921861173 8193261179 3105118548
\\  0744623799 6274956735 1885752724 8912279381 8301194912
\\
\\  9833673362 4406566430 8602139494 6395224737 1907021798
\\  6094370277 0539217176 2931767523 8467481846 7669405132
\\  0005681271 4526356082 7785771342 7577896091 7363717872
\\  1468440901 2249534301 4654958537 1050792279 6892589235
\\  4201995611 2129021960 8640344181 5981362977 4771309960
\\  5187072113 4999999837 2978049951 0597317328 1609631859
\\  5024459455 3469083026 4252230825 3344685035 2619311881
\\  7101000313 7838752886 5875332083 8142061717 7669147303
\\  5982534904 2875546873 1159562863 8823537875 9375195778
\\  1857780532 1712268066 1300192787 6611195909 2164201989
;

pub fn parsePiDigits(allocator: std.mem.Allocator, numDigits: usize) ![]u8 {
    var list = std.ArrayList(u8).init(allocator);
    errdefer list.deinit();

    for (Pi) |c| {
        if (std.ascii.isDigit(c)) {
            try list.append(c - '0');
            if (list.items.len >= numDigits) {
                break;
            }
        }
    }

    if (list.items.len < numDigits) {
        return error.TooManyDigits;
    }

    return try list.toOwnedSlice();
}


const expectEqual = std.testing.expectEqual;


test "spigot" {
    var buffer: [1000]u8 = undefined;
    try spigot_pi(&buffer, std.testing.allocator);
    for (buffer) |digit| {
        std.debug.print("{d}", .{digit});
    }
    std.debug.print("\n", .{});

    const expected = try parsePiDigits(std.testing.allocator, buffer.len);
    defer std.testing.allocator.free(expected);
    for (0..buffer.len) |i| {
        try expectEqual(expected[i], buffer[i]);
    }
}

pub fn main() !void {
    var buffer: [1000]u8 = undefined;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	const allocator = gpa.allocator();
    try spigot_pi(&buffer, allocator);
    for (buffer) |digit| {
        std.debug.print("{d}", .{digit});
    }
    std.debug.print("\n", .{});
}
