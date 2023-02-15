load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "workspace",
    with_test = False,
)

sxt_cc_component(
    name = "cpu_workspace",
    with_test = False,
    deps = [
        ":workspace",
        "//sxt/base/container:span",
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "driver",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "fold",
    impl_deps = [
        "//sxt/curve21/type:element_p3",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/curve21:multiexponentiation",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/operation:mul",
        "//sxt/scalar25/operation:muladd",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve21/operation:overload",
        "//sxt/ristretto/random:element",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/random:element",
        "//sxt/scalar25/operation:overload",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "proof_computation",
    impl_deps = [
        ":driver",
        ":proof_descriptor",
        ":workspace",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/type:element_p3",
        "//sxt/base/num:ceil_log2",
        "//sxt/ristretto/operation:compression",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/proof/transcript:transcript_utility",
        "//sxt/scalar25/type:element",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        ":cpu_driver",
        ":proof_descriptor",
        ":random_product_generation",
        "//sxt/base/num:ceil_log2",
        "//sxt/base/test:unit_test",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/ristretto/operation:compression",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/scalar25/type:element",
        "//sxt/proof/transcript:transcript",
        "//sxt/curve21/operation:overload",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/type:literal",
        "//sxt/base/error:panic",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "proof_descriptor",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/curve21/type:element_p3",
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "random_product_generation",
    impl_deps = [
        ":proof_descriptor",
        "//sxt/base/memory:alloc_utility",
        "//sxt/base/num:ceil_log2",
        "//sxt/scalar25/type:element",
        "//sxt/curve21/type:element_p3",
        "//sxt/ristretto/random:element",
        "//sxt/scalar25/random:element",
        "//sxt/base/error:assert",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/memory:alloc",
    ],
)

sxt_cc_component(
    name = "verification_computation",
    impl_deps = [
        "//sxt/base/error:assert",
        "//sxt/scalar25/operation:mul",
        "//sxt/scalar25/operation:inv",
        "//sxt/scalar25/operation:sq",
        "//sxt/scalar25/operation:neg",
        "//sxt/scalar25/operation:inner_product",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/operation:inv",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "cpu_driver",
    impl_deps = [
        ":fold",
        ":proof_descriptor",
        ":verification_computation",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/type:element_p3",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/curve21:multiexponentiation",
        "//sxt/ristretto/operation:compression",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/operation:inner_product",
        "//sxt/scalar25/operation:inv",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        ":proof_descriptor",
        ":random_product_generation",
        ":workspace",
        "//sxt/base/test:unit_test",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/scalar25/operation:inv",
        "//sxt/curve21/operation:overload",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/random:element",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/ristretto/random:element",
        "//sxt/ristretto/operation:compression",
        "//sxt/ristretto/operation:overload",
    ],
    deps = [
        ":cpu_workspace",
        ":driver",
        "//sxt/base/container:span",
    ],
)
