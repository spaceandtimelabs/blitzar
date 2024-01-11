load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "element",
    impl_deps = [
        "//sxt/base/macro:cuda_warning",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/field25/base:byte_conversion",
        "//sxt/field25/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field25/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)
