load("//bazel:sxt_component.bzl", "sxt_cc_component")

sxt_cc_component(
    name = "commitment_computation_cpu",
    deps = [
        "//sxt/base/container:span",
    ],
    impl_deps = [
        "//sxt/seqcommit/base:commitment",
        "//sxt/seqcommit/generator:base_element",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/constant:zero",
        "//sxt/curve21/ristretto:byte_conversion",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/operation:add",
    ],
    test_deps = [
        "//sxt/seqcommit/test:test_pedersen",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "commitment_computation_gpu",
    is_cuda = True,
    deps = [
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/num:divide_up",
        "//sxt/curve21/constant:zero",
        "//sxt/base/container:span",
        "//sxt/seqcommit/base:commitment",
        "//sxt/seqcommit/generator:base_element",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/ristretto:byte_conversion",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/operation:add",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/resource:managed_device_resource"
    ],
    test_deps = [
        "//sxt/seqcommit/test:test_pedersen",
        "//sxt/base/test:unit_test",
    ],
)
