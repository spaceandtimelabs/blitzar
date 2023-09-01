# load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "cuda_test")
# load("//bazel:cuda_dlink.bzl", "cuda_dlink",)

load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_objects")

# We add this -std=c++20 flag, because
# benchmarks could not be compiled without it.
# The `build --cxxopt -std=c++20` flag set in the
# `.bazelrc` file was not passed to the compiler.
# However, this flag is relevant to some modules.
def sxt_copts():
    return [
        "-std=c++20",
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
    if is_cuda:
        cuda_library(
            name = name,
            hdrs = [
                name + ".h",
            ],
            srcs = [
                name + ".cc",
            ],
            copts = sxt_copts() + [
                "-x",
                "cuda",
            ],
            alwayslink = alwayslink,
            deps = deps + impl_deps + [
                "@rules_cuda//cuda:runtime",
            ],
            visibility = ["//visibility:public"],
            **kwargs
        )
    else:
        native.cc_library(
            name = name,
            hdrs = [
                name + ".h",
            ],
            srcs = [
                name + ".cc",
            ],
            copts = sxt_copts() + copts + [
                "-fcoroutines-ts",
            ],
            implementation_deps = impl_deps,
            deps = deps,
            alwayslink = alwayslink,
            visibility = ["//visibility:public"],
            linkopts = [
                "-lm",
            ],
            **kwargs
        )
    if with_test:
        deps_p = [
            ":" + name,
        ] + deps + test_deps
        device_test_name = name + "-device.t"
        cuda_objects(
            name = device_test_name,
            deps = deps_p,
        )
        if is_cuda:
            native.cc_test(
                name = name + ".t",
                srcs = [
                    name + ".t.cc",
                ],
                copts = sxt_copts() + copts,
                deps = deps_p + [
                    ":" + device_test_name,
                ],
                visibility = ["//visibility:public"],
                **kwargs
            )
        else:
            native.cc_test(
                name = name + ".t",
                srcs = [
                    name + ".t.cc",
                ],
                copts = sxt_copts() + copts,
                deps = deps_p + [
                    ":" + device_test_name,
                ],
                visibility = ["//visibility:public"],
                **kwargs
            )
