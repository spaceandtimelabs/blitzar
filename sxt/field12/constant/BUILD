load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "one",
    with_test = False,
    deps = [
        "//sxt/field12/base:constants",
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "zero",
    with_test = False,
    deps = [
        "//sxt/field12/type:element",
    ],
)
