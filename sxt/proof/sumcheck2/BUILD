load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "verification",
    # impl_deps = [
    #     ":polynomial_utility",
    #     ":sumcheck_transcript",
    #     "//sxt/base/error:assert",
    #     "//sxt/base/log:log",
    #     "//sxt/scalar25/operation:overload",
    #     "//sxt/scalar25/type:element",
    # ],
    test_deps = [
        # ":reference_transcript",
        "//sxt/base/test:unit_test",
        # "//sxt/scalar25/type:element",
        # "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)
