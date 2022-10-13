load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "zero",
    impl_deps = [
        "//sxt/base/bit:load",
        "//sxt/base/bit:zero_equality",
        "//sxt/scalar25/base:reduce",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)
