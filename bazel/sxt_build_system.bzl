load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_objects")
load("//bazel:cuda_test.bzl", "compute_sanitize_test")

# CUDA_TOOLS_BUILD_FILE = """
# package(default_visibility = [ "//visibility:public" ])
#
# CudaToolsInfo = provider(
#   "Extra cuda toos",
#   fields = {
#     "path"
#
# def _cuda_tools_toolchain_impl(ctx):
#
#
# filegroup(
#    name = "compute_sanitizer",
#    srcs = [
#      "{cuda_path}/bin/compute-sanitizer",
#    ],
# )
# """

# def _cuda_tools_impl(ctx):
#   cuda_path = ctx.os.environ.get("CUDA_PATH").replace("\\", "/")
#   ctx.file("BUILD", CUDA_TOOLS_BUILD_FILE.format(cuda_path=cuda_path))
#   # ctx.template("cuda_test.bzl", "//bazel:cuda_test.bzl", substitutions = {
#   #   "%{cuda_path}": cuda_path,
#   # }, executable = False)
#
# cuda_tools = repository_rule(
#   implementation = _cuda_tools_impl,
#   attrs = {"toolkit_path": attr.string(mandatory = False)},
#   configure = True,
#   local = True,
#   environ = ["CUDA_PATH"],
# )

# We add this -std=c++20 flag, because
# benchmarks could not be compiled without it.
# The `build --cxxopt -std=c++20` flag set in the
# `.bazelrc` file was not passed to the compiler.
# However, this flag is relevant to some modules.
def sxt_copts():
    return [
        "-Wno-unknown-cuda-version",
    ]

def sxt_cc_component(
        name,
        copts = [],
        is_cuda = False,
        with_test = True,
        alwayslink = False,
        test_deps = [],
        deps = [],
        impl_deps = [],
        **kwargs):
    cuda_objects(
        name = name,
        hdrs = [
            name + ".h",
        ],
        srcs = [
            name + ".cc",
        ],
        copts = sxt_copts(),
        deps = deps + impl_deps + [
            "@local_cuda//:cuda_runtime_static",
        ],
        visibility = ["//visibility:public"],
        **kwargs
    )
    if with_test:
        test_lib = name + ".t-lib"
        deps_p = [
            ":" + name,
        ] + deps + test_deps
        cuda_library(
            name = test_lib,
            copts = sxt_copts() + copts,
            deps = [
                ":" + name,
            ] + test_deps + [
                "@com_github_catchorg_catch2//:catch2",
                "@com_github_catchorg_catch2//:catch2_main",
            ],
            srcs = [
                name + ".t.cc",
            ],
            alwayslink = 1,
            rdc = True,
        )
        native.cc_test(
            name = name + ".t",
            copts = sxt_copts() + copts,
            deps = [
                       ":" + test_lib,
                   ] + test_deps +
                   [
                       "@com_github_catchorg_catch2//:catch2",
                       "@com_github_catchorg_catch2//:catch2_main",
                   ],
            visibility = ["//visibility:public"],
            **kwargs
        )
        compute_sanitize_test(
          name = name + "_sanitize.t",
          data = [name + ".t"],
        )
        # native.sh_test(
        #   name = name + "_sanitize.t",
        #   srcs = [
        #     "//bazel:sanitize_test.sh",
        #   ],
        #   args = [
        #     "$(location @cuda_tools//:compute_sanitizer)", 
        #     "2",
        #   ],
        #   data = [
        #     "@cuda_tools//:compute_sanitizer",
        #   ],
        # )

def sxt_cc_library(
        name,
        copts = [],
        deps = [],
        impl_deps = [],
        **kwargs):
    cuda_objects(
        name = name,
        copts = sxt_copts(),
        deps = deps + impl_deps,
        visibility = ["//visibility:public"],
        **kwargs
    )

def sxt_cc_binary(
        name,
        srcs = [],
        copts = [],
        deps = [],
        **kwargs):
    libname = name + "-lib"
    cuda_library(
        name = libname,
        srcs = srcs,
        copts = sxt_copts() + copts,
        deps = deps,
        rdc = True,
        **kwargs
    )
    native.cc_binary(
        name = name,
        deps = [
            ":" + libname,
        ] + deps,
        **kwargs
    )
