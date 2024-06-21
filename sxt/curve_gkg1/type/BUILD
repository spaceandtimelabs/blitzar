load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "compact_element",
    with_test = False,
    deps = [
        "//sxt/fieldgk/constant:one",
        "//sxt/fieldgk/type:element",
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
        "//sxt/fieldgk/constant:one",
        "//sxt/fieldgk/random:element",
    ],
    deps = [
        ":element_affine",
        ":element_p2",
        "//sxt/base/container:span",
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/num:cmov",
        "//sxt/fieldgk/operation:cmov",
        "//sxt/fieldgk/operation:invert",
        "//sxt/fieldgk/operation:mul",
        "//sxt/fieldgk/type:element",
    ],
)

sxt_cc_component(
    name = "element_affine",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/fieldgk/type:literal",
    ],
    deps = [
        "//sxt/fieldgk/constant:one",
        "//sxt/fieldgk/constant:zero",
        "//sxt/fieldgk/type:element",
    ],
)

sxt_cc_component(
    name = "element_p2",
    impl_deps = [
        "//sxt/fieldgk/operation:invert",
        "//sxt/fieldgk/operation:mul",
        "//sxt/fieldgk/property:zero",
    ],
    test_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/fieldgk/operation:mul",
        "//sxt/fieldgk/random:element",
        "//sxt/fieldgk/type:literal",
    ],
    deps = [
        ":compact_element",
        ":operation_adl_stub",
        "//sxt/base/macro:cuda_callable",
        "//sxt/fieldgk/constant:one",
        "//sxt/fieldgk/constant:zero",
        "//sxt/fieldgk/operation:cmov",
        "//sxt/fieldgk/type:element",
    ],
)

sxt_cc_component(
    name = "operation_adl_stub",
    with_test = False,
)
