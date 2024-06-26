load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "curve",
    impl_deps = [
        "//sxt/curve_gk/constant:b",
        "//sxt/curve_gk/type:element_affine",
        "//sxt/curve_gk/type:element_p2",
        "//sxt/fieldgk/operation:add",
        "//sxt/fieldgk/operation:mul",
        "//sxt/fieldgk/operation:square",
        "//sxt/fieldgk/operation:sub",
        "//sxt/fieldgk/property:zero",
        "//sxt/fieldgk/type:element",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_gk/constant:b",
        "//sxt/curve_gk/constant:generator",
        "//sxt/curve_gk/type:element_affine",
        "//sxt/curve_gk/type:element_p2",
        "//sxt/fieldgk/constant:one",
        "//sxt/fieldgk/operation:mul",
        "//sxt/fieldgk/operation:neg",
    ],
)

sxt_cc_component(
    name = "identity",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_gk/constant:generator",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/curve_gk/type:element_affine",
        "//sxt/curve_gk/type:element_p2",
        "//sxt/fieldgk/property:zero",
    ],
)
