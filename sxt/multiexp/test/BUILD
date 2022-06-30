load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "compute_uint64_muladd",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
    impl_deps = [
        "//sxt/multiexp/base:exponent_sequence"
    ]
)

sxt_cc_component(
    name = "generate_uint64_generators",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
    impl_deps = []
)
