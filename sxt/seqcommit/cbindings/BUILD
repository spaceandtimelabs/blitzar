load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

load(
    "//bazel:cuda_dlink.bzl",
    "cuda_dlink",
)

sxt_cc_component(
    name = "pedersen_capi",
    with_test = False,
)

sxt_cc_component(
    name = "pedersen",
    deps = [
        ":pedersen_capi",
    ],
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
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static"
    ],
    alwayslink = 1,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/type:element_p3",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/seqcommit/generator:base_element",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/constant:zero"
    ],
)
