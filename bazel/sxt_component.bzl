load("@rules_cuda//cuda:defs.bzl", "cuda_library")

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
            rdc = True,
            copts = sxt_copts(),
            alwayslink = alwayslink,
            deps = deps + impl_deps,
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
            copts = sxt_copts() + copts,
            linkstatic = 1,
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
        is_cuda = True
        if is_cuda:
            cuda_library(
                name = name + "-test-lib",
                srcs = [
                    name + ".t.cc",
                ],
                rdc = True,
                copts = sxt_copts(),
                alwayslink = alwayslink,
                deps = depset(deps_p),
                visibility = ["//visibility:public"],
                **kwargs
            )
            device_test_name = name + "-device.t"
            native.cc_test(
                name = name + ".t",
                srcs = [],
                copts = sxt_copts() + copts,
                deps = [
                  ":" + name + "-test-lib",
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
                deps = deps_p,
                visibility = ["//visibility:public"],
                **kwargs
            )
