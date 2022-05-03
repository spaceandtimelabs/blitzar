load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "byte_conversion",
    is_cuda = True,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
    impl_deps = [
        "//sxt/curve21/type:element_p3",
        "//sxt/field51/type:element",

        "//sxt/curve21/ristretto:sqrt_ratio_m1",

        "//sxt/field51/operation:add",
        "//sxt/field51/operation:sub",
        "//sxt/field51/operation:mul",
        "//sxt/field51/operation:square",
        "//sxt/field51/operation:cmov",
        "//sxt/field51/operation:cneg",
        "//sxt/field51/operation:neg",
        "//sxt/field51/operation:abs",
        "//sxt/field51/operation:pow22523",

        "//sxt/field51/constant:d",
        "//sxt/field51/constant:one",
        "//sxt/field51/constant:sqrtm1",
        "//sxt/field51/constant:invsqrtamd",

        "//sxt/field51/property:sign",
        "//sxt/field51/property:zero",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "elligator",
    is_cuda = True,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
    impl_deps = [
        ":sqrt_ratio_m1",
        "//sxt/curve21/type:element_p3",
        "//sxt/field51/constant:d",
        "//sxt/field51/constant:one",
        "//sxt/field51/constant:sqrtm1",
        "//sxt/field51/operation:abs",
        "//sxt/field51/operation:add",
        "//sxt/field51/operation:cmov",
        "//sxt/field51/operation:mul",
        "//sxt/field51/operation:neg",
        "//sxt/field51/operation:sub",
        "//sxt/field51/operation:square",
        "//sxt/field51/type:element",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "point_formation",
    is_cuda = True,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
    impl_deps = [
        ":elligator",
        "//sxt/field51/type:element",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/type:element_p3",
    ],
    test_deps = [
        "//sxt/field51/type:element",
        "//sxt/curve21/property:curve",
        "//sxt/curve21/type:element_p3",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "sqrt_ratio_m1",
    is_cuda = True,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
    impl_deps = [
        "//sxt/curve21/type:element_p3",
        "//sxt/field51/constant:sqrtm1",
        "//sxt/field51/operation:abs",
        "//sxt/field51/operation:add",
        "//sxt/field51/operation:cmov",
        "//sxt/field51/operation:mul",
        "//sxt/field51/operation:pow22523",
        "//sxt/field51/operation:square",
        "//sxt/field51/operation:sub",
        "//sxt/field51/property:zero",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)
