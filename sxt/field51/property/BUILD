load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "sign",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field51/base:byte_conversion",
        "//sxt/field51/type:element",
    ],
)

sxt_cc_component(
    name = "zero",
    test_deps = [
        "//sxt/field51/type:literal",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/bit:zero_equality",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field51/base:byte_conversion",
        "//sxt/field51/type:element",
    ],
)
