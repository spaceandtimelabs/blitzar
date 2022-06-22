load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "commitment",
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "indexed_exponent_sequence",
    deps = [
        "//sxt/multiexp/base:exponent_sequence",
    ],
    with_test = False,
)

