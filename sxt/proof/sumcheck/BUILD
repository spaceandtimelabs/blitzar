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

sxt_cc_component(
    name = "polynomial_utility",
    impl_deps = [
        "//sxt/scalar25/operation:add",
        "//sxt/scalar25/operation:mul",
        "//sxt/scalar25/operation:sub",
        "//sxt/scalar25/operation:muladd",
        "//sxt/scalar25/type:element",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "verification",
    impl_deps = [
        ":polynomial_utility",
        ":transcript_utility",
        "//sxt/base/error:assert",
        "//sxt/base/log:log",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/type:element",
        "//sxt/proof/transcript:transcript_utility",
    ],
    test_deps = [
        ":transcript_utility",
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)
