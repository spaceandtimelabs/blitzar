load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

load(
    "//bazel:cuda_dlink.bzl",
    "cuda_dlink",
)

sxt_cc_component(
    name = "pedersen",
    impl_deps = [
        "//sxt/base/container:span",
        "//sxt/seqcommit/base:commitment",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/seqcommit/naive:commitment_computation_cpu",
        "//sxt/seqcommit/naive:commitment_computation_gpu",
    ],
    # alwayslink = 1,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

cuda_dlink(
    name = "pedersen_device",
    deps = [
        ":pedersen",
    ],
)
