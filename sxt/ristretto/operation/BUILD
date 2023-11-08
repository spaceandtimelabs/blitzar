load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "compression",
    impl_deps = [
        "//sxt/base/error:assert",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/curve21/type:element_p3",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "add",
    impl_deps = [
        "//sxt/ristretto/type:compressed_element",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/type:element_p3",
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
    name = "scalar_multiply",
    impl_deps = [
        ":add",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/type:element_p3",
        "//sxt/ristretto/base:byte_conversion",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "overload",
    impl_deps = [
        ":compression",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/curve21/operation:overload",
        "//sxt/curve21/type:element_p3",
    ],
    test_deps = [
        ":compression",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:overload",
        "//sxt/ristretto/type:literal",
        "//sxt/scalar25/type:literal",
    ],
)
