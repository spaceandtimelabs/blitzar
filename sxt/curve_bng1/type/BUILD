load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "compact_element",
    with_test = False,
    deps = [
        "//sxt/field25/constant:one",
        "//sxt/field25/type:element",
    ],
)

sxt_cc_component(
    name = "compressed_element",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "conversion_utility",
    test_deps = [
        "//sxt/base/container:span",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/field25/constant:one",
        "//sxt/field25/random:element",
    ],
    deps = [
        ":element_affine",
        ":element_p2",
        "//sxt/base/container:span",
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/num:cmov",
        "//sxt/field25/operation:cmov",
        "//sxt/field25/operation:invert",
        "//sxt/field25/operation:mul",
        "//sxt/field25/type:element",
    ],
)

sxt_cc_component(
    name = "element_affine",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/field25/constant:one",
        "//sxt/field25/constant:zero",
        "//sxt/field25/type:element",
    ],
)

sxt_cc_component(
    name = "element_p2",
    impl_deps = [
        "//sxt/field25/operation:invert",
        "//sxt/field25/operation:mul",
        "//sxt/field25/property:zero",
    ],
    test_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/field25/operation:mul",
        "//sxt/field25/random:element",
        "//sxt/field25/type:literal",
    ],
    deps = [
        ":compact_element",
        ":operation_adl_stub",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field25/constant:one",
        "//sxt/field25/constant:zero",
        "//sxt/field25/operation:cmov",
        "//sxt/field25/type:element",
    ],
)

sxt_cc_component(
    name = "operation_adl_stub",
    with_test = False,
)
