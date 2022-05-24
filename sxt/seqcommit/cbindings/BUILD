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
        "//sxt/base/container:span",
        "//sxt/seqcommit/base:commitment",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/seqcommit/naive:commitment_computation_cpu",
        "//sxt/seqcommit/naive:commitment_computation_gpu",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
    ],
    alwayslink = 1,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)
