load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "element",
    impl_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/scalar25/base:reduce",
        "//sxt/scalar25/type:element",
    ],
    test_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:element",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)
