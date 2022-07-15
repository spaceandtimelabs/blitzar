load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "base_element",
    impl_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve21/type:element_p3",
        "//sxt/ristretto/random:element",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/ristretto/base:byte_conversion",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "cpu_generator",
    impl_deps = [
        ":base_element",
        "//sxt/curve21/type:element_p3",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/ristretto/base:byte_conversion",
    ],
    test_deps = [
        "//sxt/seqcommit/test:test_generators",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "gpu_generator",
    impl_deps = [
        ":base_element",
        "//sxt/base/num:divide_up",
        "//sxt/curve21/type:element_p3",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/base/device:memory_utility",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/ristretto/base:byte_conversion",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/seqcommit/test:test_generators",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)
