load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "doubling_reduction",
    test_deps = [
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:overload",
        "//sxt/ristretto/type:literal",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/bit:count",
        "//sxt/base/bit:span_op",
        "//sxt/base/container:span",
        "//sxt/base/curve:element",
        "//sxt/base/error:assert",
    ],
)
