load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "hex",
)

sxt_cc_component(
    name = "divide_up",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "fast_random_number_generator",
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)
