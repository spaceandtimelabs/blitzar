load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "load",
    with_test = False,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "store",
    with_test = False,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "zero_equality",
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)
