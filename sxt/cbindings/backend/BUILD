load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "computational_backend",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/curve21/type:element_p3",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/ristretto/type:compressed_element",
    ],
)

sxt_cc_component(
    name = "gpu_backend",
    impl_deps = [
        "//sxt/base/error:assert",
        "//sxt/proof/transcript:transcript",
        "//sxt/scalar25/type:element",
        "//sxt/curve21/type:element_p3",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/ristretto/type:literal",
        "//sxt/ristretto/operation:compression",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/curve21:multiexponentiation",
        "//sxt/seqcommit/generator:precomputed_generators",
        "//sxt/proof/inner_product:proof_descriptor",
        "//sxt/proof/inner_product:proof_computation",
        "//sxt/proof/inner_product:gpu_driver",
    ],
    with_test = False,
    deps = [
        ":computational_backend",
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "cpu_backend",
    impl_deps = [
        "//sxt/proof/transcript:transcript",
        "//sxt/scalar25/type:element",
        "//sxt/curve21/type:element_p3",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/ristretto/operation:compression",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/seqcommit/generator:precomputed_generators",
        "//sxt/multiexp/curve21:multiexponentiation",
        "//sxt/proof/inner_product:proof_descriptor",
        "//sxt/proof/inner_product:proof_computation",
        "//sxt/proof/inner_product:cpu_driver",
    ],
    with_test = False,
    deps = [
        ":computational_backend",
        "//sxt/base/container:span",
    ],
)
