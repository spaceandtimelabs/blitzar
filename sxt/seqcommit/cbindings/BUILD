load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "pedersen_capi",
    with_test = False,
)

sxt_cc_component(
    name = "backend",
    impl_deps = [
        "//sxt/base/device:property",
        "//sxt/seqcommit/backend:naive_cpu_backend",
        "//sxt/seqcommit/backend:naive_gpu_backend",
        "//sxt/seqcommit/backend:pippenger_cpu_backend",
        "//sxt/seqcommit/generator:precomputed_initializer",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":pedersen_capi",
        "//sxt/seqcommit/backend:pedersen_backend",
    ],
    alwayslink = 1,
)

sxt_cc_component(
    name = "pedersen",
    impl_deps = [
        "//sxt/seqcommit/cbindings:backend",
        "//sxt/memory/management:managed_array",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
    ],
    test_deps = [
        ":backend",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/constant:zero",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/ristretto/operation:add",
        "//sxt/ristretto/operation:scalar_multiply",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/seqcommit/generator:base_element",
    ],
    deps = [
        ":pedersen_capi",
    ],
    alwayslink = 1,
)

sxt_cc_component(
    name = "get_generators",
    impl_deps = [
        ":backend",
        "//sxt/curve21/type:element_p3",
    ],
    test_deps = [
        ":backend",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/type:element_p3",
        "//sxt/seqcommit/generator:base_element",
    ],
    deps = [
        ":pedersen_capi",
    ],
    alwayslink = 1,
)

sxt_cc_component(
    name = "get_one_commit",
    impl_deps = [
        ":backend",
        "//sxt/curve21/type:element_p3",
        "//sxt/seqcommit/generator:precomputed_one_commitments",
    ],
    test_deps = [
        ":backend",
        ":get_generators",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/constant:zero",
        "//sxt/curve21/operation:add",
    ],
    deps = [
        ":pedersen_capi",
    ],
    alwayslink = 1,
)
