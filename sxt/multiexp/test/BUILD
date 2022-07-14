load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add_ints",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
    impl_deps = [
    ]
)

sxt_cc_component(
    name = "compute_uint64_muladd",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
    impl_deps = [
        "//sxt/multiexp/base:exponent_sequence"
    ]
)

sxt_cc_component(
    name = "compute_ristretto_muladd",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
    impl_deps = [
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/constant:zero",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/ristretto/type:compressed_element"
    ]
)

sxt_cc_component(
    name = "int_generation",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
    impl_deps = []
)

sxt_cc_component(
    name = "generate_ristretto_elements",
    with_test = False,
    deps = [
        "//sxt/base/container:span"
    ],
    impl_deps = [
        "//sxt/ristretto/type:compressed_element",
        "//sxt/curve21/type:element_p3",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/ristretto/random:element"
    ]
)
