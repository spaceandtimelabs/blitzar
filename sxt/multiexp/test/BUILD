load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add_ints",
    impl_deps = [
        "//sxt/base/error:panic",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "add_curve21_elements",
    impl_deps = [
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/constant:zero",
        "//sxt/curve21/operation:add",
        "//sxt/base/error:panic",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "compute_uint64_muladd",
    impl_deps = [
        "//sxt/multiexp/base:exponent_sequence",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "compute_ristretto_muladd",
    impl_deps = [
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/constant:zero",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/ristretto/type:compressed_element",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "generate_ristretto_elements",
    impl_deps = [
        "//sxt/ristretto/type:compressed_element",
        "//sxt/curve21/type:element_p3",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/ristretto/random:element",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "generate_curve21_elements",
    impl_deps = [
        "//sxt/curve21/type:element_p3",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/ristretto/random:element",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
)
