load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "sign",
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field51/type:element",
        "//sxt/field51/base:byte_conversion",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "zero",
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field51/type:element",
        "//sxt/field51/base:byte_conversion",
        "//sxt/base/bit:zero_equality",
    ],
    test_deps = [
        "//sxt/field51/type:literal",
        "//sxt/base/test:unit_test",
    ],
)
