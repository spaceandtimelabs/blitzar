load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "sign",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    with_test = False,
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field51/base:byte_conversion",
        "//sxt/field51/type:element",
    ],
)

sxt_cc_component(
    name = "zero",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field51/type:literal",
    ],
    deps = [
        "//sxt/base/bit:zero_equality",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field51/base:byte_conversion",
        "//sxt/field51/type:element",
    ],
)
