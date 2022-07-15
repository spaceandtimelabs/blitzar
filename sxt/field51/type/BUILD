load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "element",
    impl_deps = [
        "//sxt/field51/base:byte_conversion",
        "//sxt/field51/base:reduce",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "literal",
    deps = [
        ":element",
        "//sxt/base/num:hex",
        "//sxt/field51/base:byte_conversion",
    ],
)
