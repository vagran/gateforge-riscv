const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

// Implementation of BigInt which does not uses hardware multiplications and divisions.

pub const Sign = enum { positive, negative };

pub const BigInt = struct {
    allocator: Allocator,
    limbs: []u32,
    sign: Sign,

    pub fn init(allocator: Allocator, limbs: []const u32, sign: Sign) !BigInt {
        const trimmed = try trimLeadingZeros(allocator, limbs);
        const is_zero = allZero(trimmed);
        return BigInt{
            .allocator = allocator,
            .limbs = trimmed,
            .sign = if (is_zero) .positive else sign,
        };
    }

    pub fn deinit(self: BigInt) void {
        self.allocator.free(self.limbs);
    }

    fn allZero(limbs: []const u32) bool {
        for (limbs) |limb| if (limb != 0) return false;
        return true;
    }

    fn trimLeadingZeros(allocator: Allocator, limbs: []const u32) ![]u32 {
        var len = limbs.len;
        while (len > 0 and limbs[len - 1] == 0) len -= 1;
        const trimmed = try allocator.alloc(u32, if (len == 0) 1 else len);
        @memcpy(trimmed, limbs[0..trimmed.len]);
        return trimmed;
    }

    pub fn add(a: BigInt, b: BigInt) !BigInt {
        if (a.sign == b.sign) {
            const sum_limbs = try addMagnitudes(a.allocator, a.limbs, b.limbs);
            defer a.allocator.free(sum_limbs);
            return BigInt.init(a.allocator, sum_limbs, a.sign);
        }
        return subDifferentSign(a, b);
    }

    fn addMagnitudes(allocator: std.mem.Allocator, a: []const u32, b: []const u32) ![]u32 {
        const max_len = if (a.len > b.len) a.len else b.len;
        var result = try allocator.alloc(u32, max_len + 1);
        @memset(result, 0);

        var carry: u32 = 0;
        for (0..max_len) |i| {
            const a_limb = if (i < a.len) a[i] else 0;
            const b_limb = if (i < b.len) b[i] else 0;
            const sum = @as(u64, a_limb) + b_limb + carry;
            result[i] = @truncate(sum);
            carry = @truncate(sum >> 32);
        }
        result[max_len] = carry;

        return result;
    }

    pub fn sub(a: BigInt, b: BigInt) !BigInt {
        if (a.sign != b.sign) {
            const sum_limbs = try addMagnitudes(a.allocator, a.limbs, b.limbs);
            defer a.allocator.free(sum_limbs);
            return BigInt.init(a.allocator, sum_limbs, a.sign);
        }
        return subSameSign(a, b);
    }

    fn subSameSign(a: BigInt, b: BigInt) !BigInt {
        const cmp = compareMagnitude(a.limbs, b.limbs);
        if (cmp == .eq) return BigInt.init(a.allocator, &[_]u32{0}, .positive);

        var larger: []const u32 = undefined;
        var smaller: []const u32 = undefined;
        var result_sign: Sign = undefined;

        switch (cmp) {
            .gt => {
                larger = a.limbs;
                smaller = b.limbs;
                result_sign = if (a.sign == .positive) .positive else .negative;
            },
            .lt => {
                larger = b.limbs;
                smaller = a.limbs;
                result_sign = if (a.sign == .positive) .negative else .positive;
            },
            .eq => unreachable,
        }

        const diff_limbs = try subMagnitudes(a.allocator, larger, smaller);
        defer a.allocator.free(diff_limbs);
        return BigInt.init(a.allocator, diff_limbs, result_sign);
    }

    fn compareMagnitude(a: []const u32, b: []const u32) math.Order {
        if (a.len > b.len) return .gt;
        if (a.len < b.len) return .lt;
        var i: isize = @intCast(a.len - 1);
        while (i >= 0) : (i -= 1) {
            if (a[@intCast(i)] > b[@intCast(i)]) return .gt;
            if (a[@intCast(i)] < b[@intCast(i)]) return .lt;
        }
        return .eq;
    }

    fn subMagnitudes(allocator: std.mem.Allocator, a: []const u32, b: []const u32) ![]u32 {
        var result = try allocator.alloc(u32, a.len);
        var borrow: u32 = 0;

        for (0..a.len) |i| {
            const b_limb = if (i < b.len) b[i] else 0;
            const a_limb = a[i];

            // Use wrapping subtraction to avoid overflow checks
            const diff = (@as(u64, a_limb) -% @as(u64, b_limb)) -% @as(u64, borrow);

            result[i] = @truncate(diff);
            // Calculate borrow by checking if we underflowed (diff >= 0x100000000)
            borrow = @intCast((diff >> 32) & 1);
        }

        return result;
    }

    fn subDifferentSign(a: BigInt, b: BigInt) !BigInt {
        const cmp = compareMagnitude(a.limbs, b.limbs);

        // Handle equal magnitudes case first
        if (cmp == .eq) {
            return BigInt.init(a.allocator, &[_]u32{0}, .positive);
        }

        var larger_limbs: []const u32 = undefined;
        var smaller_limbs: []const u32 = undefined;
        var result_sign: Sign = undefined;

        // Determine which operand has larger magnitude and set result sign
        switch (cmp) {
            .gt => {
                larger_limbs = a.limbs;
                smaller_limbs = b.limbs;
                result_sign = a.sign;
            },
            .lt => {
                larger_limbs = b.limbs;
                smaller_limbs = a.limbs;
                result_sign = b.sign;
            },
            .eq => unreachable, // Handled above
        }

        // Perform magnitude subtraction
        const diff_limbs = try subMagnitudes(a.allocator, larger_limbs, smaller_limbs);
        defer a.allocator.free(diff_limbs);
        // Create final result with proper sign
        return BigInt.init(a.allocator, diff_limbs, result_sign);
    }

    pub fn mul(a: BigInt, b: BigInt) !BigInt {
        const product_limbs = try mulMagnitudes(a.allocator, a.limbs, b.limbs);
        const result_sign = if (a.sign == b.sign) Sign.positive else Sign.negative;
        defer a.allocator.free(product_limbs);
        return BigInt.init(a.allocator, product_limbs, result_sign);
    }

    fn mulMagnitudes(allocator: std.mem.Allocator, a: []const u32, b: []const u32) ![]u32 {
        const result_len = a.len + b.len;
        var result = try allocator.alloc(u32, result_len);
        @memset(result, 0);

        for (a, 0..) |a_limb, i| {
            var carry: u32 = 0;
            for (b, 0..) |b_limb, j| {
                const res_idx = i + j;
                if (res_idx >= result_len) continue;

                // Multiply 32-bit limbs to get 64-bit product
                const product = mul32(a_limb, b_limb);

                // Add lower 32 bits to current position with carry
                var total = @as(u64, result[res_idx]) + product.lo + carry;
                result[res_idx] = @truncate(total);
                carry = @truncate(total >> 32);

                // Add upper 32 bits to next position
                if (res_idx + 1 < result_len) {
                    total = @as(u64, result[res_idx + 1]) + product.hi + carry;
                    result[res_idx + 1] = @truncate(total);
                    carry = @truncate(total >> 32);
                }

                // Propagate remaining carry through higher limbs
                var k = res_idx + 2;
                while (carry > 0 and k < result_len) {
                    total = @as(u64, result[k]) + carry;
                    result[k] = @truncate(total);
                    carry = @truncate(total >> 32);
                    k += 1;
                }
            }
        }

        return result;
    }

    pub fn compare(a: BigInt, b: BigInt) math.Order {
        if (a.sign != b.sign) {
            return if (a.sign == .positive) .gt else .lt;
        }
        const cmp = compareMagnitude(a.limbs, b.limbs);
        return if (a.sign == .positive) cmp else switch (cmp) {
            .lt => .gt,
            .gt => .lt,
            .eq => .eq,
        };
    }

    pub fn negate(self: BigInt) !BigInt {
        const new_sign = switch (self.sign) {
            .positive => .negative,
            .negative => .positive,
        };
        return BigInt.init(self.allocator, self.limbs, new_sign);
    }

    pub fn div(dividend: BigInt, divisor: BigInt) !BigInt {
        // Handle division by zero
        if (try divisor.isZero()) return error.DivisionByZero;

        // Handle zero dividend
        if (try dividend.isZero()) return BigInt.init(dividend.allocator, &.{0}, .positive);

        // Determine result sign
        const result_sign = if (dividend.sign == divisor.sign) Sign.positive else Sign.negative;

        // Compare magnitudes
        const cmp = compareMagnitude(dividend.limbs, divisor.limbs);
        if (cmp == .lt) return BigInt.init(dividend.allocator, &.{0}, .positive);
        if (cmp == .eq) return BigInt.init(dividend.allocator, &.{1}, result_sign);

        // Single-limb divisor optimization
        if (divisor.limbs.len == 1) {
            return try divSingleLimb(dividend, divisor.limbs[0], result_sign);
        }

        // Multi-limb divisor implementation (simplified)
        return try divMultiLimb(dividend, divisor, result_sign);
    }

    fn divSingleLimb(dividend: BigInt, divisor: u32, result_sign: Sign) !BigInt {
        var quotient_limbs = try dividend.allocator.alloc(u32, dividend.limbs.len);
        defer dividend.allocator.free(quotient_limbs);

        var remainder_hi: u32 = 0;
        var i = dividend.limbs.len;
        while (i > 0) {
            i -= 1;
            const current_limb = dividend.limbs[i];

            // Use software division implementation
            const result = div64by32(remainder_hi, current_limb, divisor);
            quotient_limbs[i] = result.q;
            remainder_hi = result.r;
        }

        return BigInt.init(dividend.allocator, quotient_limbs, result_sign);
    }

    fn divMultiLimb(dividend: BigInt, divisor: BigInt, result_sign: Sign) !BigInt {
        var current = try dividend.copy();
        defer current.deinit();

        var quotient = try BigInt.init(current.allocator, &.{0}, result_sign);
        errdefer quotient.deinit();

        const divisor_bits = bitLength(divisor);

        while (true) {
            const current_bits = bitLength(current);
            if (current_bits < divisor_bits) break;

            // Calculate initial shift using precise bit length difference
            const shift = current_bits - divisor_bits;
            var last_valid_shift: i32 = -1;

            // Binary search for valid shift
            var low: i32 = 0;
            var high: i32 = @intCast(shift);
            while (low <= high) {
                const mid = (low + high) >> 1;
                const test_shift: u32 = @intCast(mid);
                var shifted_divisor = try divisor.shl(test_shift);
                defer shifted_divisor.deinit();

                switch (compareMagnitude(shifted_divisor.limbs, current.limbs)) {
                    .gt => high = mid - 1,
                    .lt, .eq => {
                        last_valid_shift = mid;
                        low = mid + 1;
                    },
                }
            }

            if (last_valid_shift == -1) break;
            const final_shift = @as(u32, @intCast(last_valid_shift));

            // Perform subtraction with exact shift
            var shifted_divisor = try divisor.shl(final_shift);
            defer shifted_divisor.deinit();

            const new_current = try current.sub(shifted_divisor);
            current.deinit();
            current = new_current;

            // Update quotient
            var shift_value = try BigInt.init(current.allocator, &.{1}, .positive);
            defer shift_value.deinit();
            const shifted_value = try shift_value.shl(final_shift);
            defer shifted_value.deinit();
            const new_quotient = try quotient.add(shifted_value);
            quotient.deinit();
            quotient = new_quotient;
        }

        return quotient;
    }

    pub fn mod(dividend: BigInt, divisor: BigInt) !BigInt {
        const quotient = try div(dividend, divisor);
        defer quotient.deinit();
        const product = try mul(quotient, divisor);
        defer product.deinit();
        return sub(dividend, product);
    }

    // Helper functions
    fn isZero(self: BigInt) !bool {
        return self.limbs.len == 1 and self.limbs[0] == 0;
    }

    pub fn copy(self: BigInt) !BigInt {
        return BigInt.init(self.allocator, self.limbs, self.sign);
    }

    /// Calculate number of significant bits in a BigInt
    fn bitLength(num: BigInt) u32 {
        if (num.limbs.len == 0) return 0;
        const top_limb = num.limbs[num.limbs.len - 1];
        return @intCast((num.limbs.len - 1) * 32 + (32 - @clz(top_limb)));
    }

    pub fn shl(self: BigInt, shift: u32) !BigInt {
        const limb_shift = shift >> 5;
        const bit_shift: u6 = @truncate(shift & 0x1F);

        var result = try self.allocator.alloc(u32, self.limbs.len + limb_shift + 1);
        defer self.allocator.free(result);

        @memset(result[0..limb_shift], 0);
        var carry: u32 = 0;
        for (self.limbs, 0..) |limb, i| {
            const shifted = (@as(u64, limb) << bit_shift) | carry;
            result[i + limb_shift] = @truncate(shifted);
            carry = @truncate(shifted >> 32);
        }
        result[self.limbs.len + limb_shift] = carry;

        return BigInt.init(self.allocator, result, self.sign);
    }
};

pub fn mul32(x: u32, y: u32) struct { hi: u32, lo: u32 } {
    var hi: u32 = 0;
    var lo: u32 = 0;
    var current_hi: u32 = 0; // High bits of x << i
    var current_lo = x; // Low bits of x << i
    var y_remaining = y;

    // Process all 32 bits of y
    for (0..32) |_| {
        // Add current shifted x if bit is set
        if (y_remaining & 1 != 0) {
            // Add current_lo to accumulator's lo with carry
            const sum_lo = @as(u64, lo) + current_lo;
            lo = @truncate(sum_lo);
            const carry: u32 = @truncate(sum_lo >> 32);

            // Add current_hi and any carry to accumulator's hi
            const sum_hi = @as(u64, hi) + current_hi + carry;
            hi = @truncate(sum_hi);
        }

        // Shift current x left by 1 bit (x <<= 1)
        const new_current_lo = current_lo << 1;
        const new_current_hi = (current_lo >> 31) | (current_hi << 1);
        current_lo = new_current_lo;
        current_hi = new_current_hi;

        // Move to next bit in y
        y_remaining >>= 1;
    }

    return .{ .hi = hi, .lo = lo };
}

pub fn div64by32(n_hi: u32, n_lo: u32, d: u32) struct { q: u32, r: u32 } {
    // Combine into 64-bit dividend and use 64-bit arithmetic
    const dividend: u64 = (@as(u64, n_hi) << 32) | n_lo;
    var q: u32 = 0;
    var rem: u32 = 0;

    // Restoring division algorithm
    for (0..64) |i| {
        // Shift remainder left and bring down next bit from dividend
        rem = @truncate((rem << 1) | ((dividend >> @intCast(63 - i)) & 1));

        q <<= 1;

        // Check if we can subtract divisor
        if (rem >= d) {
            rem -= d;
            q |= 1; // Set least significant bit
        }
    }

    return .{
        .q =q, // Quotient is lower 32 bits
        .r = @truncate(rem), // Remainder fits in 32 bits
    };
}
