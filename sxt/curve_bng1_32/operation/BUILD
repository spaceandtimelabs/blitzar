load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add",
    impl_deps = [
        ":cmov",
        "//sxt/curve_bng1_32/property:identity",
        "//sxt/curve_bng1_32/type:element_affine",
    ],
    is_cuda = True,
    test_deps = [
        ":double",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/curve_bng1_32/constant:generator",
        "//sxt/curve_bng1_32/property:curve",
        "//sxt/curve_bng1_32/property:identity",
        "//sxt/curve_bng1_32/type:element_affine",
        "//sxt/field32/random:element",
        "//sxt/field32/type:element",
    ],
    deps = [
        ":mul_by_3b",
        "//sxt/base/macro:cuda_callable",
        "//sxt/curve_bng1_32/type:element_p2",
        "//sxt/field32/operation:add",
        "//sxt/field32/operation:mul",
        "//sxt/field32/operation:sub",
    ],
)

sxt_cc_component(
    name = "cmov",
    impl_deps = [
        "//sxt/curve_bng1_32/type:element_affine",
        "//sxt/curve_bng1_32/type:element_p2",
        "//sxt/field32/operation:cmov",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_bng1_32/constant:generator",
        "//sxt/curve_bng1_32/type:element_affine",
        "//sxt/curve_bng1_32/type:element_p2",
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
        "//sxt/curve_bng1_32/property:identity",
        "//sxt/curve_bng1_32/type:element_p2",
        "//sxt/field32/operation:add",
        "//sxt/field32/operation:mul",
        "//sxt/field32/operation:square",
        "//sxt/field32/operation:sub",
        "//sxt/field32/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_bng1_32/constant:generator",
        "//sxt/curve_bng1_32/property:curve",
        "//sxt/curve_bng1_32/property:identity",
        "//sxt/curve_bng1_32/type:element_p2",
        "//sxt/field32/constant:zero",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "mul_by_3b",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field32/constant:one",
        "//sxt/field32/type:literal",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field32/operation:add",
        "//sxt/field32/type:element",
    ],
)

sxt_cc_component(
    name = "neg",
    impl_deps = [
        "//sxt/curve_bng1_32/type:element_p2",
        "//sxt/field32/operation:cmov",
        "//sxt/field32/operation:neg",
        "//sxt/field32/type:element",
    ],
    is_cuda = True,
    test_deps = [
        ":add",
        "//sxt/base/test:unit_test",
        "//sxt/curve_bng1_32/constant:generator",
        "//sxt/curve_bng1_32/type:element_p2",
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
        "//sxt/curve_bng1_32/type:element_p2",
    ],
    is_cuda = True,
    test_deps = [
        ":add",
        ":double",
        "//sxt/base/test:unit_test",
        "//sxt/curve_bng1_32/constant:generator",
        "//sxt/curve_bng1_32/type:element_p2",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)
