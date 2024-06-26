load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "b",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/fieldgk/base:montgomery",
    ],
    deps = [
        "//sxt/fieldgk/type:element",
    ],
)

sxt_cc_component(
    name = "generator",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/fieldgk/base:montgomery",
    ],
    deps = [
        "//sxt/curve_gk/type:element_affine",
        "//sxt/curve_gk/type:element_p2",
        "//sxt/fieldgk/constant:one",
        "//sxt/fieldgk/type:element",
    ],
)
