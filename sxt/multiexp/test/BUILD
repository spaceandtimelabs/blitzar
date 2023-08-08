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
    name = "curve21_arithmetic",
    impl_deps = [
        "//sxt/base/container:stack_array",
        "//sxt/base/num:constexpr_switch",
        "//sxt/base/num:ceil_log2",
        "//sxt/base/type:int",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/property:curve",
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
    name = "multiexponentiation",
    impl_deps = [
        ":curve21_arithmetic",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/operation:overload",
        "//sxt/ristretto/type:literal",
        "//sxt/ristretto/operation:compression",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/random:random_multiproduct_descriptor",
        "//sxt/multiexp/random:random_multiproduct_generation",
        "//sxt/multiexp/random:random_multiexponentiation_descriptor",
        "//sxt/multiexp/random:random_multiexponentiation_generation",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/base:exponent_sequence_utility",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/functional:function_ref",
        "//sxt/memory/management:managed_array_fwd",
    ],
)
