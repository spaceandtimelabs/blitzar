load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)
load(
    "//bazel:cuda_dlink.bzl",
    "cuda_dlink",
)

cuda_dlink(
    name = "cbindings_device",
    deps = [
        ":backend",
        ":get_generators",
        ":get_one_commit",
        ":inner_product_proof",
        ":pedersen",
    ],
)

cc_binary(
    name = "libproofs-gpu.so",
    linkopts = [
        "-Wl,--version-script=$(location :libproofs-gpu-export-map.ld)",
        "-static-libstdc++",
        "-static-libgcc",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":cbindings_device",
        ":libproofs-gpu-export-map.ld",
    ],
)

sxt_cc_component(
    name = "proofs_gpu_api",
    with_test = False,
)

sxt_cc_component(
    name = "backend",
    impl_deps = [
        "//sxt/base/device:property",
        "//sxt/base/error:assert",
        "//sxt/cbindings/backend:gpu_backend",
        "//sxt/cbindings/backend:cpu_backend",
        "//sxt/seqcommit/generator:precomputed_initializer",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":proofs_gpu_api",
        "//sxt/cbindings/backend:computational_backend",
    ],
    alwayslink = 1,
)

sxt_cc_component(
    name = "get_generators",
    impl_deps = [
        ":backend",
        "//sxt/curve21/type:element_p3",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        ":backend",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/type:element_p3",
        "//sxt/seqcommit/generator:base_element",
    ],
    deps = [
        ":proofs_gpu_api",
    ],
    alwayslink = 1,
)

sxt_cc_component(
    name = "get_one_commit",
    impl_deps = [
        ":backend",
        "//sxt/base/error:assert",
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
        ":proofs_gpu_api",
    ],
    alwayslink = 1,
)

sxt_cc_component(
    name = "pedersen",
    impl_deps = [
        ":backend",
        "//sxt/base/error:assert",
        "//sxt/memory/management:managed_array",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
        "//sxt/curve21/type:element_p3",
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
        "//sxt/base/error:assert",
    ],
    deps = [
        ":proofs_gpu_api",
    ],
    alwayslink = 1,
)

sxt_cc_component(
    name = "inner_product_proof",
    impl_deps = [
        ":backend",
        "//sxt/base/container:span",
        "//sxt/base/num:ceil_log2",
        "//sxt/curve21/type:element_p3",
        "//sxt/base/error:assert",
        "//sxt/proof/inner_product:proof_descriptor",
    ],
    test_deps = [
        ":backend",
        ":get_generators",
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:element",
        "//sxt/curve21/type:element_p3",
        "//sxt/proof/transcript:transcript",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/base/num:ceil_log2",
        "//sxt/scalar25/type:literal",
        "//sxt/scalar25/operation:overload",
        "//sxt/curve21/operation:overload",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/scalar25/random:element",
        "//sxt/base/error:assert",
    ],
    deps = [
        ":proofs_gpu_api",
    ],
    alwayslink = 1,
)
