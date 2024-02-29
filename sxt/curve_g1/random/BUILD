load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "element_p2",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve_g1/constant:b",
        "//sxt/curve_g1/property:curve",
        "//sxt/curve_g1/type:element_p2",
        "//sxt/field12/constant:one",
        "//sxt/field12/operation:add",
        "//sxt/field12/operation:mul",
        "//sxt/field12/operation:sqrt",
        "//sxt/field12/operation:square",
        "//sxt/field12/random:element",
    ],
)
