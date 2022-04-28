load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "exponent",
    with_test = False,
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/num:fast_random_number_generator",
    ],
)

sxt_cc_component(
    name = "ristretto_element",
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/field51/random:element",
        "//sxt/field51/type:element",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/ristretto:point_formation",
    ],
    test_deps = [
        "//sxt/curve21/property:curve",
    ],
)
