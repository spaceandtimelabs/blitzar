load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add",
    impl_deps = [
        ":cmov",
        "//sxt/curve_gk/property:identity",
        "//sxt/curve_gk/type:element_affine",
    ],
    is_cuda = True,
    test_deps = [
        ":double",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/curve_gk/constant:generator",
        "//sxt/curve_gk/property:curve",
        "//sxt/curve_gk/property:identity",
        "//sxt/curve_gk/type:element_affine",
        "//sxt/fieldgk/random:element",
        "//sxt/fieldgk/type:element",
    ],
    deps = [
        ":mul_by_3b",
        "//sxt/base/macro:cuda_callable",
        "//sxt/curve_gk/type:element_p2",
        "//sxt/fieldgk/operation:add",
        "//sxt/fieldgk/operation:mul",
        "//sxt/fieldgk/operation:sub",
    ],
)

sxt_cc_component(
    name = "cmov",
    impl_deps = [
        "//sxt/curve_gk/type:element_affine",
        "//sxt/curve_gk/type:element_p2",
        "//sxt/fieldgk/operation:cmov",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_gk/constant:generator",
        "//sxt/curve_gk/type:element_affine",
        "//sxt/curve_gk/type:element_p2",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "double",
    impl_deps = [
        ":cmov",
        ":mul_by_3b",
        "//sxt/curve_gk/property:identity",
        "//sxt/curve_gk/type:element_p2",
        "//sxt/fieldgk/operation:add",
        "//sxt/fieldgk/operation:mul",
        "//sxt/fieldgk/operation:square",
        "//sxt/fieldgk/operation:sub",
        "//sxt/fieldgk/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_gk/constant:generator",
        "//sxt/curve_gk/property:curve",
        "//sxt/curve_gk/property:identity",
        "//sxt/curve_gk/type:element_p2",
        "//sxt/fieldgk/constant:zero",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "mul_by_3b",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/fieldgk/constant:one",
        "//sxt/fieldgk/type:element",
        "//sxt/fieldgk/type:literal",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/fieldgk/operation:mul",
        "//sxt/fieldgk/type:element",
    ],
)

sxt_cc_component(
    name = "neg",
    impl_deps = [
        "//sxt/curve_gk/type:element_p2",
        "//sxt/fieldgk/operation:cmov",
        "//sxt/fieldgk/operation:neg",
        "//sxt/fieldgk/type:element",
    ],
    is_cuda = True,
    test_deps = [
        ":add",
        "//sxt/base/test:unit_test",
        "//sxt/curve_gk/constant:generator",
        "//sxt/curve_gk/type:element_p2",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "scalar_multiply",
    impl_deps = [
        ":add",
        ":double",
        "//sxt/curve_gk/type:element_p2",
    ],
    is_cuda = True,
    test_deps = [
        ":add",
        ":double",
        "//sxt/base/test:unit_test",
        "//sxt/curve_gk/constant:generator",
        "//sxt/curve_gk/type:element_p2",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)
