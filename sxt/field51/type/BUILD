load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "element",
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
    impl_deps = [
        "//sxt/field51/base:byte_conversion",
        "//sxt/field51/base:reduce",
    ],
)

sxt_cc_component(
    name = "literal",
    deps = [
        ":element",
        "//sxt/field51/base:byte_conversion",
        "//sxt/base/num:hex",
    ],
)
