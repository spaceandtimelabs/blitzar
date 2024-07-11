load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "element_affine",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_bng1/property:curve",
    ],
    deps = [
        ":element_p2",
        "//sxt/base/macro:cuda_callable",
        "//sxt/curve_bng1/type:conversion_utility",
        "//sxt/curve_bng1/type:element_affine",
        "//sxt/curve_bng1/type:element_p2",
    ],
)

sxt_cc_component(
    name = "element_p2",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_bng1/property:curve",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve_bng1/constant:generator",
        "//sxt/curve_bng1/operation:scalar_multiply",
        "//sxt/curve_bng1/type:element_p2",
    ],
)
