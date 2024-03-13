load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "element_p2",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_g1/property:curve",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve_g1/constant:generator",
        "//sxt/curve_g1/operation:scalar_multiply",
        "//sxt/curve_g1/type:element_p2",
    ],
)
