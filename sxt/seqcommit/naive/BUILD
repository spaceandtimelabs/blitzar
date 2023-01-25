load("//bazel:sxt_component.bzl", "sxt_cc_component")

sxt_cc_component(
    name = "commitment_computation_gpu",
    impl_deps = [
        "//sxt/base/device:memory_utility",
        "//sxt/base/num:divide_up",
        "//sxt/execution/base:stream",
        "//sxt/curve21/constant:zero",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/operation:add",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/base/error:assert",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/type:element_p3",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/seqcommit/generator:base_element",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)
