load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
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
        "//sxt/multiexp/ristretto:pippenger_multiproduct_solver",
        "//sxt/multiexp/ristretto:precomputed_p3_input_accessor",
        "//sxt/multiexp/ristretto:multiexponentiation_cpu_driver",
        "//sxt/multiexp/pippenger:multiexponentiation",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/operation:mul",
        "//sxt/scalar25/operation:muladd",
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
    name = "workspace",
    with_test = False,
)
