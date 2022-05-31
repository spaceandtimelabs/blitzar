load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "base_element",
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
    is_cuda = True,
    impl_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve21/random:ristretto_element",
        "//sxt/curve21/type:element_p3",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "cpu_generator",
    deps = [
        "//sxt/base/container:span",
    ],
    impl_deps = [
        ":base_element",
        "//sxt/curve21/type:element_p3",
        "//sxt/seqcommit/base:commitment",
        "//sxt/curve21/ristretto:byte_conversion"
    ],
    test_deps = [
        "//sxt/seqcommit/test:test_generators",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "gpu_generator",
    deps = [
        "//sxt/base/container:span",
    ],
    is_cuda = True,
    impl_deps = [
        ":base_element",
        "//sxt/base/num:divide_up",
        "//sxt/curve21/type:element_p3",
        "//sxt/seqcommit/base:commitment",
        "//sxt/base/device:memory_utility",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/curve21/ristretto:byte_conversion"
    ],
    test_deps = [
        "//sxt/seqcommit/test:test_generators",
        "//sxt/base/test:unit_test",
    ],
)
