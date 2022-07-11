load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "pedersen_backend",
    with_test = False,
    deps = [
        "//sxt/ristretto/type:compressed_element",
        "//sxt/base/container:span",
        "//sxt/seqcommit/base:indexed_exponent_sequence"
    ]
)

sxt_cc_component(
    name = "naive_cpu_backend",
    with_test = False,
    deps = [
        ":pedersen_backend",
        "//sxt/base/container:span"
    ],
    impl_deps = [
        "//sxt/ristretto/type:compressed_element",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
        "//sxt/seqcommit/naive:commitment_computation_cpu",
        "//sxt/seqcommit/generator:cpu_generator"
    ]
)

sxt_cc_component(
    name = "naive_gpu_backend",
    with_test = False,
    deps = [
        ":pedersen_backend",
        "//sxt/base/container:span"
    ],
    impl_deps = [
        "//sxt/memory/management:managed_array",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
        "//sxt/seqcommit/naive:commitment_computation_gpu",
        "//sxt/seqcommit/generator:gpu_generator"
    ]
)

sxt_cc_component(
    name = "pippenger_cpu_backend",
    with_test = False,
    deps = [
        ":pedersen_backend",
        "//sxt/base/container:span"
    ],
    impl_deps = [
        "//sxt/memory/management:managed_array",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
        "//sxt/seqcommit/generator:cpu_generator",
        "//sxt/multiexp/ristretto:multiexponentiation_cpu_driver",
        "//sxt/multiexp/pippenger:multiexponentiation",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/seqcommit/naive:commitment_computation_cpu"
    ]
)
