""

load("//cuda/private:action_names.bzl", "ACTION_NAMES")
load("//cuda/private:actions/compile.bzl", "compile")
load("//cuda/private:cuda_helper.bzl", "cuda_helper")
load("//cuda/private:toolchain.bzl", "find_cuda_toolkit")

def device_link(
        ctx,
        cuda_toolchain,
        cc_toolchain,
        objects,
        common,
        pic = False,
        rdc = False,
        dlto = False):
    """Perform device link, return a dlink-ed object file.

    Notes:
        Compilation is carried out during device linking, which involves the embeeding of the fatbin into the resulting object `File`.

    Args:
        ctx: A [context object](https://bazel.build/rules/lib/ctx).
        cuda_toolchain: A `platform_common.ToolchainInfo` of a cuda toolchain, Can be obtained with `find_cuda_toolchain(ctx)`.
        cc_toolchain: A `CcToolchainInfo`. Can be obtained with `find_cpp_toolchain(ctx)`.
        objects: A `depset` of `File`s to be device linked.
        common: A cuda common object. Can be obtained with `cuda_helper.create_common(ctx)`
        pic: Whether the `objects` are compiled for position independent code.
        rdc: Whether the `objects` are device linked for relocatable device code.
        dlto: Whether the device link time optimization is enabled.

    Returns:
        An deviced linked object `File`.
    """
    cuda_feature_config = cuda_helper.configure_features(ctx, cuda_toolchain, requested_features = [ACTION_NAMES.device_link])
    if cuda_helper.is_enabled(cuda_feature_config, "supports_compiler_device_link"):
        return _compiler_device_link(ctx, cuda_toolchain, cc_toolchain, cuda_feature_config, objects, common, pic = pic, rdc = rdc, dlto = dlto)
    elif cuda_helper.is_enabled(cuda_feature_config, "supports_wrapper_device_link"):
        return _wrapper_device_link(ctx, cuda_toolchain, cc_toolchain, objects, common, pic = pic, rdc = rdc, dlto = dlto)
    else:
        fail("toolchain must be configured to enable feature supports_compiler_device_link or supports_wrapper_device_link.")

def _compiler_device_link(
        ctx,
        cuda_toolchain,
        cc_toolchain,
        cuda_feature_config,
        objects,
        common,
        pic = False,
        rdc = False,
        dlto = False):
    """perform compiler supported native device link, return a dlink-ed object file"""
    if not rdc:
        fail("device link is only meaningful on building relocatable device code")

    actions = ctx.actions
    host_compiler = cc_toolchain.compiler_executable
    cuda_compiler = cuda_toolchain.compiler_executable

    artifact_category_name = cuda_helper.get_artifact_category_from_action(ACTION_NAMES.device_link, pic, rdc)
    basename = ctx.attr.name + "_dlink"
    filename = cuda_helper.get_artifact_name(cuda_toolchain, artifact_category_name, basename)

    obj_file = actions.declare_file("_objs/{}/{}".format(ctx.attr.name, filename))

    var = cuda_helper.create_device_link_variables(
        cuda_toolchain,
        cuda_feature_config,
        common.cuda_archs_info,
        output_file = obj_file.path,
        host_compiler = host_compiler,
        host_compile_flags = common.host_compile_flags,
        user_link_flags = common.link_flags,
        use_pic = pic,
    )
    cmd = cuda_helper.get_command_line(cuda_feature_config, ACTION_NAMES.device_link, var)
    env = cuda_helper.get_environment_variables(cuda_feature_config, ACTION_NAMES.device_link, var)
    args = actions.args()
    args.add_all(cmd)
    args.add_all(objects)

    actions.run(
        executable = cuda_compiler,
        arguments = [args],
        outputs = [obj_file],
        inputs = depset(transitive = [objects, cc_toolchain.all_files, cuda_toolchain.all_files]),
        env = env,
        mnemonic = "CudaDeviceLink",
        progress_message = "Device linking %{output}",
    )
    return obj_file

def _wrapper_device_link(
        ctx,
        cuda_toolchain,
        cc_toolchain,
        objects,
        common,
        pic = False,
        rdc = False,
        dlto = False):
    """perform bazel macro supported device link, return a dlink-ed object file"""
    if not rdc:
        fail("device link is only meaningful on building relocatable device code")

    cuda_toolkit = find_cuda_toolkit(ctx)

    actions = ctx.actions
    pic_suffix = "_pic" if pic else ""

    # Device-link to cubins for each gpu architecture. The stage1 compiled PTX is embeded in the object files.
    # We don't need to do any thing about it, presumably.
    register_h = None
    cubins = []
    images = []
    obj_args = actions.args()
    obj_args.add_all(objects)
    if len(common.cuda_archs_info.arch_specs) == 0:
        fail('cuda toolchain "' + cuda_toolchain.name + '" is configured to enable feature supports_wrapper_device_link,' +
             " at least one cuda arch must be specified.")
    for arch_spec in common.cuda_archs_info.arch_specs:
        for stage2_arch in arch_spec.stage2_archs:
            if stage2_arch.gpu:
                arch = "sm_" + stage2_arch.arch
            elif stage2_arch.lto:
                arch = "lto_" + stage2_arch.arch
            else:
                # PTX is JIT-linked at runtime
                continue

            register_h = ctx.actions.declare_file("_dlink{suffix}/{0}/{0}_register_{1}.h".format(ctx.attr.name, arch, suffix = pic_suffix))
            cubin = ctx.actions.declare_file("_dlink{suffix}/{0}/{0}_{1}.cubin".format(ctx.attr.name, arch, suffix = pic_suffix))
            ctx.actions.run(
                outputs = [register_h, cubin],
                inputs = objects,
                executable = cuda_toolkit.nvlink,
                arguments = [
                    "--arch=" + arch,
                    "--register-link-binaries=" + register_h.path,
                    "--output-file=" + cubin.path,
                    obj_args,
                ],
                mnemonic = "nvlink",
            )
            cubins.append(cubin)
            images.append("--image=profile={},file={}".format(arch, cubin.path))

    # Generate fatbin header from all cubins.
    fatbin = ctx.actions.declare_file("_dlink{suffix}/{0}/{0}.fatbin".format(ctx.attr.name, suffix = pic_suffix))
    fatbin_h = ctx.actions.declare_file("_dlink{suffix}/{0}/{0}_fatbin.h".format(ctx.attr.name, suffix = pic_suffix))

    arguments = [
        "-64",
        "--cmdline=--compile-only",
        "--link",
        "--compress-all",
        "--create=" + fatbin.path,
        "--embedded-fatbin=" + fatbin_h.path,
    ]
    bin2c = cuda_toolkit.bin2c
    if (cuda_toolkit.version_major, cuda_toolkit.version_minor) <= (10, 1):
        arguments.append("--bin2c-path=%s" % bin2c.dirname)
    ctx.actions.run(
        outputs = [fatbin, fatbin_h],
        inputs = cubins,
        executable = cuda_toolkit.fatbinary,
        arguments = arguments + images,
        tools = [bin2c],
        mnemonic = "fatbinary",
    )

    # Generate the source file #including the headers generated above.
    fatbin_c = ctx.actions.declare_file("_dlink{suffix}/{0}/{0}.cu".format(ctx.attr.name, suffix = pic_suffix))
    ctx.actions.expand_template(
        output = fatbin_c,
        template = cuda_toolkit.link_stub,
        substitutions = {
            "REGISTERLINKBINARYFILE": '"{}"'.format(register_h.short_path),
            "FATBINFILE": '"{}"'.format(fatbin_h.short_path),
        },
    )

    # cc_common.compile will cause file conflict for pic and non-pic objects,
    # and it accepts only one set of src files. But pic fatbin_c and non-pic
    # fatbin_c have different compilation trajectories. This makes me feel bad.
    # Just avoid cc_common.compile at all.
    compile_common = cuda_helper.create_common_info(
        # this is useless
        cuda_archs_info = common.cuda_archs_info,
        headers = [fatbin_h],
        defines = [
            # Silence warning about including internal header.
            "__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__",
            # Macros that need to be defined starting with CUDA 10.
            "__NV_EXTRA_INITIALIZATION=",
            "__NV_EXTRA_FINALIZATION=",
        ],
        # TODO: avoid the hardcode path
        includes = common.includes + ["external/local_cuda/cuda/include"],
        system_includes = common.system_includes,
        quote_includes = common.quote_includes,
        # suppress cuda mode as c++ mode
        compile_flags = ["-x", "c++"],
        host_compile_flags = common.host_compile_flags,
    )
    ret = compile(ctx, cuda_toolchain, cc_toolchain, srcs = [fatbin_c], common = compile_common, pic = pic, rdc = rdc, _prefix = "_objs/_dlink")
    return ret[0]
