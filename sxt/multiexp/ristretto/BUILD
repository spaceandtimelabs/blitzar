load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "compressed_input_accessor",
    impl_deps = [
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/ristretto/type:compressed_element",
    ],
    with_test = False,
    deps = [
        ":input_accessor",
    ],
)

sxt_cc_component(
    name = "uncompressed_input_accessor",
    impl_deps = [
        "//sxt/curve21/type:element_p3",
    ],
    with_test = False,
    deps = [
        ":input_accessor",
    ],
)

sxt_cc_component(
    name = "precomputed_p3_input_accessor",
    impl_deps = [
        "//sxt/base/error:assert",
    ],
    with_test = False,
    deps = [
        ":input_accessor",
        "//sxt/base/container:span",
        "//sxt/curve21/type:element_p3",
    ],
)

sxt_cc_component(
    name = "input_accessor",
    with_test = False,
)

sxt_cc_component(
    name = "multiexponentiation_cpu_driver",
    impl_deps = [
        ":compressed_input_accessor",
        ":uncompressed_input_accessor",
        ":naive_multiproduct_solver",
        "//sxt/base/bit:count",
        "//sxt/base/bit:span_op",
        "//sxt/base/container:blob_array",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/curve21:doubling_reduction",
        "//sxt/multiexp/index:index_table",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/type:element_p1p1",
        "//sxt/curve21/type:conversion_utility",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve21/type:element_p3",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/random:random_multiexponentiation_descriptor",
        "//sxt/multiexp/random:random_multiexponentiation_generation",
        "//sxt/multiexp/pippenger:multiexponentiation",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/multiexp/test:compute_ristretto_muladd",
        "//sxt/multiexp/test:generate_ristretto_elements",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/ristretto/random:element",
        "//sxt/ristretto/operation:scalar_multiply",
        "//sxt/base/container:span",
        "//sxt/ristretto/operation:add",
    ],
    deps = [
        "//sxt/multiexp/pippenger:driver",
    ],
)

sxt_cc_component(
    name = "multiproduct_solver",
    with_test = False,
    deps = [
        "//sxt/memory/management:managed_array_fwd",
    ],
)

sxt_cc_component(
    name = "naive_multiproduct_solver",
    impl_deps = [
        "//sxt/base/container:span",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/operation:add",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/index:index_table",
        "//sxt/base/error:assert",
    ],
    with_test = False,
    deps = [
        ":multiproduct_solver",
    ],
)

sxt_cc_component(
    name = "pippenger_multiproduct_solver",
    impl_deps = [
        "//sxt/base/container:span",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/operation:add",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/curve21:multiproduct_cpu_driver",
        "//sxt/multiexp/index:index_table",
        "//sxt/multiexp/pippenger_multiprod:multiproduct",
    ],
    with_test = False,
    deps = [
        ":multiproduct_solver",
    ],
)
