load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "base_element",
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
    impl_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve21/random:ristretto_element",
        "//sxt/curve21/type:element_p3",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "commitment",
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)
