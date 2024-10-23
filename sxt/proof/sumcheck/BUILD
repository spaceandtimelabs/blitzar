load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "transcript_utility",
    impl_deps = [
        "//sxt/scalar25/type:element",
        "//sxt/proof/transcript:transcript_utility",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/proof/transcript",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)
