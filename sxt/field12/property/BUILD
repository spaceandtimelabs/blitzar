load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "zero",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/bit:zero_equality",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field12/base:byte_conversion",
        "//sxt/field12/type:element",
    ],
)
