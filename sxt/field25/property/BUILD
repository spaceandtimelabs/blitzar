load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "lexicographically_largest",
    impl_deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/field12/base:reduce",
        "//sxt/field12/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:one",
        "//sxt/field12/constant:zero",
        "//sxt/field12/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
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
