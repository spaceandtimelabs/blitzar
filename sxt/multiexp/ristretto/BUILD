load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "multiexponentiation_cpu_driver",
    deps = [
        "//sxt/multiexp/pippenger:driver"
    ],
    impl_deps = [
        "//sxt/base/bit:count",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/index:index_table",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/type:element_p1p1",
        "//sxt/curve21/type:conversion_utility",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/ristretto/type:compressed_element"
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/random:random_multiexponentiation_descriptor",
        "//sxt/multiexp/random:random_multiexponentiation_generation",
        "//sxt/multiexp/pippenger:multiexponentiation",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/multiexp/test:compute_ristretto_muladd",
        "//sxt/multiexp/test:generate_ristretto_elements",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/ristretto/random:element",
        "//sxt/ristretto/operation:scalar_multiply",
        "//sxt/base/container:span",
        "//sxt/ristretto/operation:add"
    ],
)
