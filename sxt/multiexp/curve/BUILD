load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "accumulator",
    test_deps = [
        "//sxt/algorithm/base:reducer",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/operation:overload",
        "//sxt/curve21/type:element_p3",
    ],
    deps = [
        "//sxt/base/curve:element",
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "doubling_reduction",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/operation:overload",
        "//sxt/ristretto/type:literal",
    ],
    deps = [
        "//sxt/base/bit:count",
        "//sxt/base/bit:span_op",
        "//sxt/base/container:span",
        "//sxt/base/curve:element",
        "//sxt/base/error:assert",
    ],
)

sxt_cc_component(
    name = "multiexponentiation",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/type:element_p3",
        "//sxt/execution/schedule:scheduler",
        "//sxt/multiexp/test:multiexponentiation",
    ],
    deps = [
        ":multiexponentiation_cpu_driver",
        ":multiproduct",
        ":multiproducts_combination",
        ":pippenger_multiproduct_solver",
        "//sxt/base/container:blob_array",
        "//sxt/base/container:span",
        "//sxt/base/device:event",
        "//sxt/base/device:event_utility",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/iterator:index_range",
        "//sxt/base/num:divide_up",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/execution/device:device_viewable",
        "//sxt/execution/device:for_each",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:device_resource",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/bucket_method:multiexponentiation",
        "//sxt/multiexp/bucket_method:multiexponentiation2",
        "//sxt/multiexp/pippenger:multiexponentiation",
        "//sxt/multiexp/pippenger:multiproduct_decomposition_gpu",
    ],
)

sxt_cc_component(
    name = "multiexponentiation_cpu_driver",
    test_deps = [
        ":naive_multiproduct_solver",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/type:element_p3",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/pippenger:multiexponentiation",
        "//sxt/multiexp/test:multiexponentiation",
    ],
    deps = [
        ":multiproduct_solver",
        ":multiproducts_combination",
        "//sxt/base/container:blob_array",
        "//sxt/base/curve:element",
        "//sxt/base/error:assert",
        "//sxt/execution/async:future",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/pippenger:driver",
    ],
)

sxt_cc_component(
    name = "multiproduct_bitset_operator",
    with_test = False,
    deps = [
        "//sxt/base/curve:element",
    ],
)

sxt_cc_component(
    name = "multiproduct",
    with_test = False,
    deps = [
        ":accumulator",
        "//sxt/algorithm/base:gather_mapper",
        "//sxt/base/container:span",
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/type:raw_stream",
        "//sxt/execution/async:future",
        "//sxt/execution/async:future_fwd",
        "//sxt/multiexp/multiproduct_gpu:multiproduct",
    ],
)

sxt_cc_component(
    name = "multiproduct_cpu_driver",
    test_deps = [
        "//sxt/base/container:span",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/type:element_p3",
        "//sxt/multiexp/index:index_table",
        "//sxt/multiexp/pippenger_multiprod:multiproduct",
        "//sxt/multiexp/random:random_multiproduct_descriptor",
        "//sxt/multiexp/random:random_multiproduct_generation",
        "//sxt/multiexp/test:curve21_arithmetic",
        "//sxt/ristretto/random:element",
    ],
    deps = [
        ":multiproduct_bitset_operator",
        "//sxt/base/container:span_void",
        "//sxt/base/curve:element",
        "//sxt/base/error:assert",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/bitset_multiprod:multiproduct",
        "//sxt/multiexp/bitset_multiprod:value_cache",
        "//sxt/multiexp/bitset_multiprod:value_cache_utility",
        "//sxt/multiexp/index:clump2_descriptor",
        "//sxt/multiexp/index:clump2_marker_utility",
        "//sxt/multiexp/pippenger_multiprod:driver",
    ],
)

sxt_cc_component(
    name = "multiproduct_solver",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/curve:element",
        "//sxt/execution/async:future_fwd",
        "//sxt/memory/management:managed_array_fwd",
    ],
)

sxt_cc_component(
    name = "multiproducts_combination",
    with_test = False,
    deps = [
        ":doubling_reduction",
        "//sxt/base/bit:span_op",
        "//sxt/base/container:blob_array",
        "//sxt/base/container:span",
        "//sxt/base/container:stack_array",
        "//sxt/base/curve:element",
        "//sxt/base/error:assert",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/management:managed_array_fwd",
        "//sxt/multiexp/base:exponent_sequence",
    ],
)

sxt_cc_component(
    name = "naive_multiproduct_solver",
    with_test = False,
    deps = [
        ":multiproduct_solver",
        "//sxt/base/curve:element",
        "//sxt/execution/async:future",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/base:generator_utility",
        "//sxt/multiexp/index:index_table",
    ],
)

sxt_cc_component(
    name = "pippenger_multiproduct_solver",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/operation:overload",
        "//sxt/curve21/type:element_p3",
        "//sxt/ristretto/type:literal",
    ],
    deps = [
        ":multiproduct_cpu_driver",
        ":multiproduct_solver",
        "//sxt/base/container:blob_array",
        "//sxt/base/curve:element",
        "//sxt/execution/async:future",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/base:generator_utility",
        "//sxt/multiexp/index:index_table",
        "//sxt/multiexp/index:reindex",
        "//sxt/multiexp/pippenger_multiprod:active_offset",
        "//sxt/multiexp/pippenger_multiprod:multiproduct",
    ],
)
