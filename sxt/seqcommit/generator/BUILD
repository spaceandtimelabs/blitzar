load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "precomputed_generators",
    impl_deps = [
        "//sxt/curve21/type:element_p3",
        "//sxt/seqcommit/generator:cpu_generator",
        "//sxt/seqcommit/generator:gpu_generator",
    ],
    test_deps = [
        ":base_element",
        "//sxt/curve21/type:element_p3",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "precomputed_one_commitments",
    impl_deps = [
        ":cpu_one_commitments",
        "//sxt/curve21/type:element_p3",
    ],
    test_deps = [
        ":cpu_one_commitments",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/type:element_p3",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "precomputed_initializer",
    impl_deps = [
        ":precomputed_generators",
        ":precomputed_one_commitments",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [],
)

sxt_cc_component(
    name = "cpu_one_commitments",
    impl_deps = [
        ":precomputed_generators",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/constant:zero",
    ],
    test_deps = [
        ":cpu_generator",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/constant:zero",
        "//sxt/curve21/type:element_p3",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
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
        "//sxt/base/device:memory_utility",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
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
