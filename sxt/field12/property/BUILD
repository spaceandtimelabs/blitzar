load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "zero",
    test_deps = [
        "//sxt/field12/type:literal",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field12/constant:zero",
        "//sxt/field12/type:element",
    ],
)
