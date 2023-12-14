load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add",
    test_deps = [
        "//sxt/base/field:example_element",
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:zero",
        "//sxt/field12/type:element",
    ],
    deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/base/field:element",
        "//sxt/base/field:subtract_p",
        "//sxt/base/macro:cuda_callable",
    ],
)
