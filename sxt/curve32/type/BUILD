load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "operation_adl_stub",
    with_test = False,
)

sxt_cc_component(
    name = "byte_conversion",
    impl_deps = [
        ":element_p3",
        "//sxt/field32/base:byte_conversion",
        "//sxt/field32/type:element",
        "//sxt/field32/operation:invert",
        "//sxt/field32/operation:add",
        "//sxt/field32/operation:neg",
        "//sxt/field32/operation:cmov",
        "//sxt/field32/operation:pow22523",
        "//sxt/field32/operation:sub",
        "//sxt/field32/operation:mul",
        "//sxt/field32/operation:sq",
        "//sxt/field32/property:sign",
        "//sxt/field32/property:zero",
        "//sxt/field32/constant:one",
        "//sxt/field32/constant:sqrtm1",
        "//sxt/field32/constant:d",
    ],
    with_test = False,
)

sxt_cc_component(
    name = "point_formation",
    impl_deps = [
        ":byte_conversion",
        ":cofactor_utility",
        ":element_p3",
        "//sxt/curve32/base:elligate",
        "//sxt/curve32/base:mont_ed_conversion",
        "//sxt/field32/type:element",
        "//sxt/field32/base:byte_conversion",
        "//sxt/field32/operation:neg",
        "//sxt/field32/operation:cmov",
        "//sxt/field32/operation:mul",
        "//sxt/field32/property:sign",
        "//sxt/field32/constant:one",
    ],
    test_deps = [
        ":element_p3",
        "//sxt/base/test:unit_test",
    ],
    with_test = False,
)

sxt_cc_component(
    name = "double_impl",
    impl_deps = [
        "//sxt/field32/type:element",
        "//sxt/field32/operation:add",
        "//sxt/field32/operation:sub",
        "//sxt/field32/operation:sq",
    ],
    is_cuda = True,
    with_test = False,
    deps = [
        ":conversion_utility",
        ":element_p2",
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "cofactor_utility",
    impl_deps = [
        ":double_impl",
        ":conversion_utility",
        ":element_p1p1",
        ":element_p2",
    ],
    with_test = False,
)

sxt_cc_component(
    name = "conversion_utility",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":element_cached",
        ":element_p1p1",
        ":element_p2",
        ":element_p3",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field32/constant:d",
        "//sxt/field32/operation:add",
        "//sxt/field32/operation:mul",
        "//sxt/field32/operation:sub",
    ],
)

sxt_cc_component(
    name = "element_cached",
    with_test = False,
    deps = [
        "//sxt/field32/type:element",
    ],
)

sxt_cc_component(
    name = "element_p1p1",
    with_test = False,
    deps = [
        "//sxt/field32/type:element",
    ],
)

sxt_cc_component(
    name = "element_p2",
    with_test = False,
    deps = [
        "//sxt/field32/type:element",
    ],
)

sxt_cc_component(
    name = "element_p3",
    impl_deps = [
        "//sxt/field32/operation:mul",
    ],
    with_test = False,
    deps = [
        ":operation_adl_stub",
        "//sxt/field32/constant:one",
        "//sxt/field32/constant:zero",
        "//sxt/field32/type:element",
    ],
)

sxt_cc_component(
    name = "literal",
    with_test = False,
    deps = [
        ":element_p3",
        ":point_formation",
        "//sxt/base/type:literal",
    ],
)
