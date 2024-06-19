load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "lexicographically_largest",
    impl_deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/fieldgk/base:reduce",
        "//sxt/fieldgk/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/fieldgk/base:montgomery",
        "//sxt/fieldgk/constant:one",
        "//sxt/fieldgk/constant:zero",
        "//sxt/fieldgk/type:element",
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
        "//sxt/fieldgk/base:byte_conversion",
        "//sxt/fieldgk/type:element",
    ],
)
