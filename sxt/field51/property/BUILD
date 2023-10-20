load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "sign",
    with_test = False,
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
