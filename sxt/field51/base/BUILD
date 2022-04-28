load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "byte_conversion",
    is_cuda = True,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
    impl_deps = [
        ":reduce",
        "//sxt/base/bit:load",
        "//sxt/base/bit:store",
    ],
)

sxt_cc_component(
    name = "reduce",
    deps = [
        "//sxt/base/type:int",
        "//sxt/base/macro:cuda_callable",
    ],
    test_deps = [
    ],
)
