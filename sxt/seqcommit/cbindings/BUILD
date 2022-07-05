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
    name = "pedersen_backend",
    with_test = False,
    deps = [
        "//sxt/seqcommit/base:commitment",
        "//sxt/base/container:span",
        "//sxt/seqcommit/base:indexed_exponent_sequence"
    ]
)

sxt_cc_component(
    name = "pedersen_cpu_backend",
    with_test = False,
    deps = [
        ":pedersen_backend",
        "//sxt/base/container:span"
    ],
    impl_deps = [
        "//sxt/seqcommit/base:commitment",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
        "//sxt/seqcommit/naive:commitment_computation_cpu",
        "//sxt/seqcommit/generator:cpu_generator"
    ]
)

sxt_cc_component(
    name = "pedersen_gpu_backend",
    with_test = False,
    deps = [
        ":pedersen_backend",
        "//sxt/base/container:span"
    ],
    impl_deps = [
        "//sxt/memory/management:managed_array",
        "//sxt/seqcommit/base:commitment",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
        "//sxt/seqcommit/naive:commitment_computation_gpu",
        "//sxt/seqcommit/generator:gpu_generator"
    ]
)

sxt_cc_component(
    name = "pedersen",
    deps = [
        ":pedersen_capi",
    ],
    impl_deps = [
        ":pedersen_backend",
        ":pedersen_cpu_backend",
        ":pedersen_gpu_backend",
        "//sxt/base/container:span",
        "//sxt/seqcommit/base:commitment",
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
        "//sxt/seqcommit/base:commitment",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/seqcommit/generator:base_element",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/constant:zero"
    ],
)
