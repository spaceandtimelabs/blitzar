load(
    "//bazel:sxt_build_system.bzl",
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
        "//sxt/base/type:literal",
        "//sxt/field51/base:byte_conversion",
    ],
)
