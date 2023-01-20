load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "transcript",
    impl_deps = [
        "//sxt/base/error:assert",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":strobe128",
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "keccakf",
    impl_deps = [],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [],
)

sxt_cc_component(
    name = "strobe128",
    impl_deps = [
        ":keccakf",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "transcript_primitive",
    impl_deps = [],
    with_test = False,
    deps = [],
)

sxt_cc_component(
    name = "transcript_utility",
    impl_deps = [
        "//sxt/scalar25/operation:reduce",
        "//sxt/scalar25/type:element",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/scalar25/operation:reduce",
        "//sxt/scalar25/type:element",
    ],
    deps = [
        ":transcript",
        ":transcript_primitive",
        "//sxt/base/container:span",
    ],
)
