load(
    "@rules_cuda//cuda:defs.bzl",
    "cuda_library",
)
load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

cuda_library(
    name = "cbindings_device",
    rdc = 1,
    deps = [
        ":backend",
        ":get_generators",
        ":get_one_commit",
        ":inner_product_proof",
        ":pedersen",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "libblitzar.so",
    linkopts = [
        "-Wl,--version-script=$(location :libblitzar-export-map.ld)",
        "-lm",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":cbindings_device",
        ":libblitzar-export-map.ld",
    ] + select({
        "//bazel:portable_glibc_build": [
            "//bazel:glibc_version_lib",
        ],
        "//conditions:default": [],
    }),
)

sxt_cc_component(
    name = "blitzar_api",
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
        ":blitzar_api",
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
        ":blitzar_api",
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
        "//sxt/curve21/operation:add",
        "//sxt/curve21/type:element_p3",
    ],
    deps = [
        ":blitzar_api",
    ],
    alwayslink = 1,
)

sxt_cc_component(
    name = "pedersen",
    impl_deps = [
        ":backend",
        "//sxt/base/error:assert",
        "//sxt/curve_g1/type:compressed_element",
        "//sxt/curve_g1/type:conversion_utility",
        "//sxt/curve_g1/type:element_affine",
        "//sxt/curve_g1/type:element_p2",
        "//sxt/curve21/type:element_p3",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/ristretto/type:compressed_element",
    ],
    test_deps = [
        ":backend",
        "//sxt/base/error:assert",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve_g1/constant:generator",
        "//sxt/curve_g1/operation:add",
        "//sxt/curve_g1/operation:compression",
        "//sxt/curve_g1/operation:scalar_multiply",
        "//sxt/curve_g1/type:compressed_element",
        "//sxt/curve_g1/type:conversion_utility",
        "//sxt/curve_g1/type:element_affine",
        "//sxt/memory/management:managed_array",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/ristretto/operation:add",
        "//sxt/ristretto/operation:overload",
        "//sxt/ristretto/operation:scalar_multiply",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/seqcommit/generator:base_element",
    ],
    deps = [
        ":blitzar_api",
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
        "//sxt/base/error:assert",
        "//sxt/base/num:ceil_log2",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:overload",
        "//sxt/curve21/type:element_p3",
        "//sxt/proof/transcript",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/random:element",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        ":blitzar_api",
    ],
    alwayslink = 1,
)
