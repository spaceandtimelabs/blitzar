load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "conversion_utility",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":element_affine",
        ":element_p2",
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/num:cmov",
        "//sxt/field12/operation:cmov",
        "//sxt/field12/operation:invert",
        "//sxt/field12/operation:mul",
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "element_affine",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/field12/constant:one",
        "//sxt/field12/constant:zero",
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "element_p2",
    impl_deps = [
        "//sxt/field12/operation:mul",
        "//sxt/field12/property:zero",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/operation:mul",
    ],
    deps = [
        ":operation_adl_stub",
        "//sxt/field12/constant:one",
        "//sxt/field12/constant:zero",
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "operation_adl_stub",
    with_test = False,
)
