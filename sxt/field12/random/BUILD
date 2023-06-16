load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "element",
    impl_deps = [
        "//sxt/base/macro:cuda_warning",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/field12/base:byte_conversion",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field12/type:element",
    ],
)
