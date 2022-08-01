load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "pedersen_backend",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/curve21/type:element_p3",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
    ],
)

sxt_cc_component(
    name = "naive_cpu_backend",
    impl_deps = [
        "//sxt/curve21/type:element_p3",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
        "//sxt/seqcommit/naive:commitment_computation_cpu",
        "//sxt/seqcommit/generator:cpu_generator",
        "//sxt/seqcommit/generator:precomputed_generators",
    ],
    with_test = False,
    deps = [
        ":pedersen_backend",
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "naive_gpu_backend",
    impl_deps = [
        "//sxt/curve21/type:element_p3",
        "//sxt/memory/management:managed_array",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
        "//sxt/seqcommit/naive:commitment_computation_gpu",
        "//sxt/seqcommit/generator:gpu_generator",
        "//sxt/seqcommit/generator:precomputed_generators",
    ],
    with_test = False,
    deps = [
        ":pedersen_backend",
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "pippenger_cpu_backend",
    impl_deps = [
        "//sxt/curve21/type:element_p3",
        "//sxt/memory/management:managed_array",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
        "//sxt/seqcommit/generator:cpu_generator",
        "//sxt/seqcommit/generator:precomputed_generators",
        "//sxt/multiexp/ristretto:precomputed_p3_input_accessor",
        "//sxt/multiexp/ristretto:multiexponentiation_cpu_driver",
        "//sxt/multiexp/ristretto:pippenger_multiproduct_solver",
        "//sxt/multiexp/pippenger:multiexponentiation",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/seqcommit/naive:commitment_computation_cpu",
    ],
    with_test = False,
    deps = [
        ":pedersen_backend",
        "//sxt/base/container:span",
    ],
)
