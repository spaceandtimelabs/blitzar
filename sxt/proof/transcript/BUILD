load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "transcript",
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
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)
