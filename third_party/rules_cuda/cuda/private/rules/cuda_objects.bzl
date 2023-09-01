load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("//cuda/private:cuda_helper.bzl", "cuda_helper")
load("//cuda/private:providers.bzl", "CudaInfo")
load("//cuda/private:toolchain.bzl", "find_cuda_toolchain", "use_cpp_toolchain", "use_cuda_toolchain")
load("//cuda/private:actions/compile.bzl", "compile")
load("//cuda/private:rules/common.bzl", "ALLOW_CUDA_HDRS", "ALLOW_CUDA_SRCS")

def _cuda_objects_impl(ctx):
    attr = ctx.attr
    cuda_helper.check_srcs_extensions(ctx, ALLOW_CUDA_SRCS + ALLOW_CUDA_HDRS, "cuda_object")

    cc_toolchain = find_cpp_toolchain(ctx)
    cuda_toolchain = find_cuda_toolchain(ctx)

    common = cuda_helper.create_common(ctx)

    # flatten first, so that non-unique basenames can be properly deduplicated
    src_files = []
    for src in ctx.attr.srcs:
        src_files.extend(src[DefaultInfo].files.to_list())

    # outputs
    objects = depset(compile(ctx, cuda_toolchain, cc_toolchain, src_files, common, pic = False, rdc = False))
    rdc_objects = depset(compile(ctx, cuda_toolchain, cc_toolchain, src_files, common, pic = False, rdc = True))
    pic_objects = depset(compile(ctx, cuda_toolchain, cc_toolchain, src_files, common, pic = True, rdc = False))
    rdc_pic_objects = depset(compile(ctx, cuda_toolchain, cc_toolchain, src_files, common, pic = True, rdc = True))

    compilation_ctx = cc_common.create_compilation_context(
        headers = common.headers,
        includes = depset(common.includes),
        system_includes = depset(common.system_includes),
        quote_includes = depset(common.quote_includes),
        defines = depset(common.host_defines),
        local_defines = depset(common.host_local_defines),
    )

    return [
        # default output is only enabled for rdc_objects, otherwise, when you build with
        #
        # > bazel build //cuda_objects/that/needs/rdc/...
        #
        # compiling errors might be trigger due to objects and pic_objects been built if srcs require device link
        DefaultInfo(
            files = depset(transitive = [
                # objects,
                # pic_objects,
                rdc_objects,
                # rdc_pic_objects,
            ]),
        ),
        OutputGroupInfo(
            objects = objects,
            pic_objects = pic_objects,
            rdc_objects = rdc_objects,
            rdc_pic_objects = rdc_pic_objects,
        ),
        CcInfo(
            compilation_context = compilation_ctx,
        ),
        cuda_helper.create_cuda_info(
            defines = depset(common.defines),
            objects = objects,
            pic_objects = pic_objects,
            rdc_objects = rdc_objects,
            rdc_pic_objects = rdc_pic_objects,
        ),
    ]

cuda_objects = rule(
    doc = """This rule produces incomplete object files that can only be consumed by `cuda_library`. It is created for relocatable device
code and device link time optimization source files.""",
    implementation = _cuda_objects_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = ALLOW_CUDA_SRCS + ALLOW_CUDA_HDRS),
        "hdrs": attr.label_list(allow_files = ALLOW_CUDA_HDRS),
        "deps": attr.label_list(providers = [[CcInfo], [CudaInfo]]),
        "includes": attr.string_list(doc = "List of include dirs to be added to the compile line."),
        # host_* attrs will be passed transitively to cc_* and cuda_* targets
        "host_copts": attr.string_list(doc = "Add these options to the CUDA host compilation command."),
        "host_defines": attr.string_list(doc = "List of defines to add to the compile line."),
        "host_local_defines": attr.string_list(doc = "List of defines to add to the compile line, but only apply to this rule."),
        # non-host attrs will be passed transitively to cuda_* targets only.
        "copts": attr.string_list(doc = "Add these options to the CUDA device compilation command."),
        "defines": attr.string_list(doc = "List of defines to add to the compile line."),
        "local_defines": attr.string_list(doc = "List of defines to add to the compile line, but only apply to this rule."),
        "ptxasopts": attr.string_list(doc = "Add these flags to the ptxas command."),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),  # legacy behaviour
        "_default_cuda_copts": attr.label(default = "//cuda:copts"),
        "_default_host_copts": attr.label(default = "//cuda:host_copts"),
        "_default_cuda_archs": attr.label(default = "//cuda:archs"),
    },
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain() + use_cuda_toolchain(),
    provides = [DefaultInfo, OutputGroupInfo, CcInfo, CudaInfo],
)
