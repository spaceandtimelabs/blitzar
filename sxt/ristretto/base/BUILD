load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "byte_conversion",
    impl_deps = [
        ":sqrt_ratio_m1",
        "//sxt/curve32/type:element_p3",
        "//sxt/field32/type:element",
        "//sxt/field32/operation:add",
        "//sxt/field32/operation:sub",
        "//sxt/field32/operation:mul",
        "//sxt/field32/operation:sq",
        "//sxt/field32/operation:cmov",
        "//sxt/field32/operation:cneg",
        "//sxt/field32/operation:neg",
        "//sxt/field32/operation:abs",
        "//sxt/field32/operation:pow22523",
        "//sxt/field32/constant:d",
        "//sxt/field32/constant:one",
        "//sxt/field32/constant:sqrtm1",
        "//sxt/field32/constant:invsqrtamd",
        "//sxt/field32/property:sign",
        "//sxt/field32/property:zero",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "elligator",
    impl_deps = [
        "//sxt/ristretto/base:sqrt_ratio_m1",
        "//sxt/curve32/type:element_p3",
        "//sxt/field32/constant:d",
        "//sxt/field32/constant:one",
        "//sxt/field32/constant:sqrtm1",
        "//sxt/field32/operation:abs",
        "//sxt/field32/operation:add",
        "//sxt/field32/operation:cmov",
        "//sxt/field32/operation:mul",
        "//sxt/field32/operation:neg",
        "//sxt/field32/operation:sub",
        "//sxt/field32/operation:sq",
        "//sxt/field32/type:element",
    ],
    is_cuda = True,
    with_test = False,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "sqrt_ratio_m1",
    impl_deps = [
        "//sxt/curve32/type:element_p3",
        "//sxt/field32/constant:sqrtm1",
        "//sxt/field32/operation:abs",
        "//sxt/field32/operation:add",
        "//sxt/field32/operation:cmov",
        "//sxt/field32/operation:mul",
        "//sxt/field32/operation:pow22523",
        "//sxt/field32/operation:sq",
        "//sxt/field32/operation:sub",
        "//sxt/field32/property:zero",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    with_test = False,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "point_formation",
    impl_deps = [
        ":elligator",
        "//sxt/field32/type:element",
        "//sxt/curve32/operation:add",
        "//sxt/curve32/type:element_p3",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve32/property:curve",
        "//sxt/curve32/type:element_p3",
        "//sxt/field32/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)
