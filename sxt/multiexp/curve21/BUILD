load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "doubling_reduction",
    impl_deps = [
        "//sxt/base/bit:count",
        "//sxt/base/bit:span_op",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        "//sxt/curve21/operation:overload",
        "//sxt/ristretto/type:literal",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "multiproduct_bitset_operator",
    with_test = False,
    deps = [
        "//sxt/curve21/operation:add",
        "//sxt/curve21/type:element_p3",
    ],
)

sxt_cc_component(
    name = "multiexponentiation",
    impl_deps = [
        ":multiexponentiation_cpu_driver",
        ":multiproducts_combination",
        ":pippenger_multiproduct_solver",
        "//sxt/base/container:blob_array",
        "//sxt/curve21/operation:accumulator",
        "//sxt/memory/management:managed_array",
        "//sxt/curve21/type:element_p3",
        "//sxt/multiexp/pippenger:multiexponentiation",
        "//sxt/multiexp/pippenger:multiproduct_decomposition",
        "//sxt/execution/async:future",
        "//sxt/multiexp/multiproduct_gpu:multiproduct",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/memory/management:managed_array",
        "//sxt/curve21/type:element_p3",
        "//sxt/base/test:unit_test",
        "//sxt/multiexp/test:multiexponentiation",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/execution/async:future",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/execution/async:future_fwd",
        "//sxt/memory/management:managed_array_fwd",
    ],
)

sxt_cc_component(
    name = "multiexponentiation_cpu_driver",
    impl_deps = [
        ":multiproduct_solver",
        ":multiproducts_combination",
        "//sxt/base/container:blob_array",
        "//sxt/memory/management:managed_array",
        "//sxt/execution/async:future",
        "//sxt/curve21/type:element_p3",
    ],
    test_deps = [
        ":naive_multiproduct_solver",
        "//sxt/curve21/type:element_p3",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/pippenger:multiexponentiation",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/execution/async:future",
        "//sxt/multiexp/test:multiexponentiation",
    ],
    deps = [
        "//sxt/multiexp/pippenger:driver",
    ],
)

sxt_cc_component(
    name = "multiproduct_cpu_driver",
    impl_deps = [
        ":multiproduct_bitset_operator",
        "//sxt/curve21/constant:zero",
        "//sxt/curve21/operation:add",
        "//sxt/base/container:span_void",
        "//sxt/curve21/type:element_p3",
        "//sxt/multiexp/index:clump2_descriptor",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/bitset_multiprod:value_cache",
        "//sxt/multiexp/bitset_multiprod:value_cache_utility",
        "//sxt/multiexp/bitset_multiprod:multiproduct",
        "//sxt/multiexp/index:clump2_marker_utility",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        "//sxt/base/container:span",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/type:element_p3",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/pippenger_multiprod:multiproduct",
        "//sxt/multiexp/index:index_table",
        "//sxt/multiexp/test:curve21_arithmetic",
        "//sxt/ristretto/random:element",
        "//sxt/multiexp/random:random_multiproduct_descriptor",
        "//sxt/multiexp/random:random_multiproduct_generation",
    ],
    deps = [
        "//sxt/multiexp/pippenger_multiprod:driver",
    ],
)

sxt_cc_component(
    name = "multiproduct_solver",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/execution/async:future_fwd",
        "//sxt/memory/management:managed_array_fwd",
    ],
)

sxt_cc_component(
    name = "multiproducts_combination",
    impl_deps = [
        ":doubling_reduction",
        "//sxt/base/bit:span_op",
        "//sxt/base/error:assert",
        "//sxt/base/container:blob_array",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/constant:zero",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "naive_multiproduct_solver",
    impl_deps = [
        "//sxt/execution/async:future",
        "//sxt/memory/management:managed_array",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/operation:add",
        "//sxt/multiexp/base:generator_utility",
        "//sxt/multiexp/index:index_table",
    ],
    with_test = False,
    deps = [
        ":multiproduct_solver",
    ],
)

sxt_cc_component(
    name = "pippenger_multiproduct_solver",
    impl_deps = [
        ":multiproduct_cpu_driver",
        "//sxt/base/container:blob_array",
        "//sxt/execution/async:future",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/pippenger_multiprod:multiproduct",
        "//sxt/multiexp/pippenger_multiprod:active_offset",
        "//sxt/curve21/type:element_p3",
        "//sxt/multiexp/index:index_table",
        "//sxt/multiexp/index:reindex",
        "//sxt/multiexp/base:generator_utility",
    ],
    test_deps = [
        "//sxt/base/container:blob_array",
        "//sxt/multiexp/index:index_table",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/constant:zero",
        "//sxt/curve21/operation:overload",
        "//sxt/ristretto/type:literal",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/execution/async:future",
    ],
    deps = [
        ":multiproduct_solver",
    ],
)
