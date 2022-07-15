load("//bazel:sxt_component.bzl", "sxt_cc_component")

sxt_cc_component(
    name = "commitment_computation_cpu",
    impl_deps = [
        "//sxt/ristretto/type:compressed_element",
        "//sxt/seqcommit/generator:base_element",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/constant:zero",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/operation:add",
    ],
    test_deps = [
        "//sxt/seqcommit/test:test_pedersen",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "commitment_computation_gpu",
    impl_deps = [
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/num:divide_up",
        "//sxt/curve21/constant:zero",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/seqcommit/generator:base_element",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/operation:add",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/ristretto/base:byte_conversion",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/seqcommit/test:test_pedersen",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)
