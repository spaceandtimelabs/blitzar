load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "block_size",
    with_test = False,
)

sxt_cc_component(
    name = "launch",
    with_test = False,
    deps = [
        ":block_size",
    ],
)

sxt_cc_component(
    name = "kernel_dims",
    with_test = False,
    deps = [
        ":block_size",
    ],
)
