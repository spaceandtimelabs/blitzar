load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "computational_backend",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/cbindings/base:curve_id",
        "//sxt/curve21/type:element_p3",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/pippenger2:partition_table_accessor_base",
        "//sxt/ristretto/type:compressed_element",
    ],
)

sxt_cc_component(
    name = "computational_backend_utility",
    impl_deps = [
        "//sxt/base/error:assert",
        "//sxt/base/num:divide_up",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "gpu_backend",
    impl_deps = [
        ":computational_backend_utility",
        "//sxt/base/error:assert",
        "//sxt/base/num:divide_up",
        "//sxt/proof/transcript:transcript",
        "//sxt/scalar25/type:element",
        "//sxt/cbindings/base:curve_id_utility",
        "//sxt/curve_bng1/operation:add",
        "//sxt/curve_bng1/operation:double",
        "//sxt/curve_bng1/operation:neg",
        "//sxt/curve_bng1/type:conversion_utility",
        "//sxt/curve_bng1/type:element_affine",
        "//sxt/curve_bng1/type:element_p2",
        "//sxt/curve_g1/operation:add",
        "//sxt/curve_g1/operation:compression",
        "//sxt/curve_g1/operation:double",
        "//sxt/curve_g1/operation:neg",
        "//sxt/curve_g1/type:compressed_element",
        "//sxt/curve_g1/type:element_p2",
        "//sxt/curve_gk/operation:add",
        "//sxt/curve_gk/operation:double",
        "//sxt/curve_gk/operation:neg",
        "//sxt/curve_gk/type:conversion_utility",
        "//sxt/curve_gk/type:element_affine",
        "//sxt/curve_gk/type:element_p2",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/type:element_p3",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/ristretto/type:literal",
        "//sxt/ristretto/operation:compression",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/curve:multiexponentiation",
        "//sxt/multiexp/pippenger2:in_memory_partition_table_accessor_utility",
        "//sxt/multiexp/pippenger2:multiexponentiation",
        "//sxt/multiexp/pippenger2:variable_length_multiexponentiation",
        "//sxt/seqcommit/generator:precomputed_generators",
        "//sxt/proof/inner_product:proof_descriptor",
        "//sxt/proof/inner_product:proof_computation",
        "//sxt/proof/inner_product:gpu_driver",
    ],
    with_test = False,
    deps = [
        ":computational_backend",
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "cpu_backend",
    impl_deps = [
        ":computational_backend_utility",
        "//sxt/base/error:panic",
        "//sxt/base/num:round_up",
        "//sxt/cbindings/base:curve_id_utility",
        "//sxt/proof/transcript:transcript",
        "//sxt/scalar25/type:element",
        "//sxt/curve_bng1/operation:add",
        "//sxt/curve_bng1/operation:double",
        "//sxt/curve_bng1/operation:neg",
        "//sxt/curve_bng1/type:conversion_utility",
        "//sxt/curve_bng1/type:element_affine",
        "//sxt/curve_bng1/type:element_p2",
        "//sxt/curve_g1/operation:add",
        "//sxt/curve_g1/operation:compression",
        "//sxt/curve_g1/operation:double",
        "//sxt/curve_g1/operation:neg",
        "//sxt/curve_g1/type:compressed_element",
        "//sxt/curve_g1/type:element_p2",
        "//sxt/curve_gk/operation:add",
        "//sxt/curve_gk/operation:double",
        "//sxt/curve_gk/operation:neg",
        "//sxt/curve_gk/type:conversion_utility",
        "//sxt/curve_gk/type:element_affine",
        "//sxt/curve_gk/type:element_p2",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/type:element_p3",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/pippenger2:in_memory_partition_table_accessor_utility",
        "//sxt/multiexp/pippenger2:multiexponentiation",
        "//sxt/multiexp/pippenger2:variable_length_multiexponentiation",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/ristretto/operation:compression",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/seqcommit/generator:precomputed_generators",
        "//sxt/multiexp/curve:multiexponentiation",
        "//sxt/proof/inner_product:proof_descriptor",
        "//sxt/proof/inner_product:proof_computation",
        "//sxt/proof/inner_product:cpu_driver",
    ],
    with_test = False,
    deps = [
        ":computational_backend",
        "//sxt/base/container:span",
    ],
)
