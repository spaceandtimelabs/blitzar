load("//cuda/private:providers.bzl", "CudaToolchainConfigInfo", "CudaToolkitInfo")
load("//cuda/private:toolchain_config_lib.bzl", "config_helper")

def _cuda_toolchain_impl(ctx):
    cuda_toolchain_config = ctx.attr.toolchain_config[CudaToolchainConfigInfo]
    selectables_info = config_helper.collect_selectables_info(cuda_toolchain_config.action_configs + cuda_toolchain_config.features)
    must_have_selectables = []
    for name in must_have_selectables:
        if not config_helper.is_configured(selectables_info, name):
            fail(name, "is not configured (not exists) in the provided toolchain_config")

    artifact_name_patterns = {}
    for pattern in cuda_toolchain_config.artifact_name_patterns:
        artifact_name_patterns[pattern.category_name] = pattern

    return [
        platform_common.ToolchainInfo(
            name = ctx.label.name,
            compiler_executable = ctx.attr.compiler_executable,
            all_files = ctx.attr.compiler_files.files if ctx.attr.compiler_files else depset(),
            selectables_info = selectables_info,
            artifact_name_patterns = artifact_name_patterns,
            cuda_toolkit = cuda_toolchain_config.cuda_toolkit,
        ),
    ]

cuda_toolchain = rule(
    doc = """This rule consumes a `CudaToolchainConfigInfo` and provides a `platform_common.ToolchainInfo`, a.k.a, the CUDA Toolchain.""",
    implementation = _cuda_toolchain_impl,
    attrs = {
        "toolchain_config": attr.label(
            mandatory = True,
            providers = [CudaToolchainConfigInfo],
            doc = "A target that provides a `CudaToolchainConfigInfo`.",
        ),
        "compiler_executable": attr.string(mandatory = True, doc = "The path of the main executable of this toolchain."),
        "compiler_files": attr.label(allow_files = True, cfg = "exec", doc = "The set of files that are needed when compiling using this toolchain."),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),
    },
)

CPP_TOOLCHAIN_TYPE = "@bazel_tools//tools/cpp:toolchain_type"
CUDA_TOOLCHAIN_TYPE = "//cuda:toolchain_type"

# buildifier: disable=unused-variable
def use_cpp_toolchain(mandatory = True):
    """Helper to depend on the C++ toolchain.

    Notes:
        Copied from [toolchain_utils.bzl](https://github.com/bazelbuild/bazel/blob/ac48e65f70/tools/cpp/toolchain_utils.bzl#L53-L72)
    """
    return [CPP_TOOLCHAIN_TYPE]

def use_cuda_toolchain():
    """Helper to depend on the CUDA toolchain."""
    return [CUDA_TOOLCHAIN_TYPE]

def find_cuda_toolchain(ctx):
    """Helper to get the cuda toolchain from context object.

    Args:
        ctx: The rule context for which to find a toolchain.

    Returns:
        A `platform_common.ToolchainInfo` that wraps around the necessary information of a cuda toolchain.
    """
    return ctx.toolchains[CUDA_TOOLCHAIN_TYPE]

def find_cuda_toolkit(ctx):
    """Finds the CUDA toolchain.

    Args:
        ctx: The rule context for which to find a toolchain.

    Returns:
        A CudaToolkitInfo.
    """
    return ctx.toolchains[CUDA_TOOLCHAIN_TYPE].cuda_toolkit[CudaToolkitInfo]

# buildifier: disable=unnamed-macro
def register_detected_cuda_toolchains():
    """Helper to register the automatically detected CUDA toolchain(s).

User can setup their own toolchain if needed and ignore the detected ones by not calling this macro.
"""
    native.register_toolchains(
        "@local_cuda//toolchain:nvcc-local-toolchain",
        "@local_cuda//toolchain/clang:clang-local-toolchain",
    )
