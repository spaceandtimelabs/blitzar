load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "identity",
    with_test = False,
    deps = [
        "//sxt/curve21/type:element_cached",
        "//sxt/curve21/type:element_p3",
        "//sxt/field51/constant:one",
        "//sxt/field51/constant:zero",
    ],
)
