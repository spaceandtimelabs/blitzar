load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add",
    impl_deps = [
        ":cmov",
        ":mul_by_3b",
        "//sxt/curve_g1/property:identity",
        "//sxt/curve_g1/type:element_affine",
        "//sxt/curve_g1/type:element_p2",
        "//sxt/field12/operation:add",
        "//sxt/field12/operation:mul",
        "//sxt/field12/operation:sub",
    ],
    is_cuda = True,
    test_deps = [
        ":double",
        "//sxt/base/test:unit_test",
        "//sxt/curve_g1/constant:generator",
        "//sxt/curve_g1/constant:identity",
        "//sxt/curve_g1/property:curve",
        "//sxt/curve_g1/property:identity",
        "//sxt/curve_g1/type:element_p2",
        "//sxt/field12/operation:mul",
        "//sxt/field12/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "cmov",
    impl_deps = [
        "//sxt/curve_g1/type:element_p2",
        "//sxt/field12/operation:cmov",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_g1/constant:generator",
        "//sxt/curve_g1/type:element_p2",
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
        "//sxt/curve_g1/property:identity",
        "//sxt/curve_g1/type:element_p2",
        "//sxt/field12/operation:add",
        "//sxt/field12/operation:mul",
        "//sxt/field12/operation:square",
        "//sxt/field12/operation:sub",
        "//sxt/field12/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_g1/constant:generator",
        "//sxt/curve_g1/property:curve",
        "//sxt/curve_g1/property:identity",
        "//sxt/curve_g1/type:element_p2",
        "//sxt/field12/constant:zero",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "mul_by_3b",
    impl_deps = [
        "//sxt/field12/operation:add",
        "//sxt/field12/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:one",
        "//sxt/field12/type:element",
        "//sxt/field12/type:literal",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "neg",
    impl_deps = [
        "//sxt/curve_g1/type:element_p2",
        "//sxt/field12/operation:neg",
    ],
    is_cuda = True,
    test_deps = [
        ":add",
        "//sxt/base/test:unit_test",
        "//sxt/curve_g1/constant:generator",
        "//sxt/curve_g1/type:element_p2",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "sub",
    impl_deps = [
        ":add",
        ":neg",
        "//sxt/curve_g1/type:element_p2",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_g1/constant:generator",
        "//sxt/curve_g1/type:element_p2",
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
        "//sxt/curve_g1/type:element_p2",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_g1/constant:generator",
        "//sxt/curve_g1/type:element_p2",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)
