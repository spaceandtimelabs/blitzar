load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "pedersen_capi",
    with_test = False,
)

sxt_cc_component(
    name = "pedersen",
    impl_deps = [
        "//sxt/seqcommit/backend:pedersen_backend",
        "//sxt/seqcommit/backend:naive_cpu_backend",
        "//sxt/seqcommit/backend:naive_gpu_backend",
        "//sxt/base/container:span",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/memory/management:managed_array",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
        "//sxt/seqcommit/naive:commitment_computation_cpu",
        "//sxt/seqcommit/naive:commitment_computation_gpu",
        "//sxt/seqcommit/backend:pippenger_cpu_backend",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/ristretto/operation:add",
        "//sxt/ristretto/random:element",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/seqcommit/generator:base_element",
        "//sxt/ristretto/operation:scalar_multiply",
    ],
    deps = [
        ":pedersen_capi",
    ],
    alwayslink = 1,
)
