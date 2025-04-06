const std = @import("std");
const CrossTarget = @import("std").zig.CrossTarget;
const Target = @import("std").Target;
const Feature = @import("std").Target.Cpu.Feature;


pub fn build(b: *std.Build) !void {
    const features = Target.riscv.Feature;
    var disabled_features = Feature.Set.empty;
    var enabled_features = Feature.Set.empty;

    const compressed = b.option(bool, "compressed", "Compile for RV32IC") orelse false;

    // disable all CPU extensions
    disabled_features.addFeature(@intFromEnum(features.a));
    disabled_features.addFeature(@intFromEnum(features.d));
    disabled_features.addFeature(@intFromEnum(features.e));
    disabled_features.addFeature(@intFromEnum(features.f));
    disabled_features.addFeature(@intFromEnum(features.m));

    if (compressed) {
        enabled_features.addFeature(@intFromEnum(features.c));
    } else {
        disabled_features.addFeature(@intFromEnum(features.c));
    }

    const target = b.resolveTargetQuery(.{
        .cpu_arch = Target.Cpu.Arch.riscv32,
        .os_tag = Target.Os.Tag.freestanding,
        .abi = Target.Abi.none,
        .cpu_model = .{ .explicit = &std.Target.riscv.cpu.generic_rv32},
        .cpu_features_sub = disabled_features,
        .cpu_features_add = enabled_features
    });

    const name = b.option([]const u8, "name", "Firmware variant name") orelse unreachable;

    const firmwareExeName = try std.mem.concat(b.allocator, u8, &[_][]const u8{
        "firmware_", name,  if (compressed) "_C" else ""});
    defer b.allocator.free(firmwareExeName);

    const firmwareMainSrcName = try std.mem.concat(b.allocator, u8, &[_][]const u8{
        "main_", name, ".zig" });
    defer b.allocator.free(firmwareExeName);

    const firmwareBinName = try std.mem.concat(b.allocator, u8, &[_][]const u8{
        "firmware_", name, if (compressed) "_C" else "", ".bin" });
    defer b.allocator.free(firmwareExeName);

    const exe = b.addExecutable(.{
        .name = firmwareExeName,
        .root_source_file = b.path(firmwareMainSrcName),
        .target = target,
        .optimize = .ReleaseSmall,
    });

    exe.setLinkerScript(b.path("linker.ld"));

    const bin = b.addObjCopy(exe.getEmittedBin(), .{
        .format = .bin,
    });
    bin.step.dependOn(&exe.step);

    const copy_bin = b.addInstallBinFile(bin.getOutput(), firmwareBinName);
    b.default_step.dependOn(&copy_bin.step);

    b.installArtifact(exe);
}
