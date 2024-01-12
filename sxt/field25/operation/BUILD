load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field25/constant:zero",
    ],
    deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field25/base:constants",
        "//sxt/field25/base:subtract_p",
        "//sxt/field25/type:element",
    ],
)

sxt_cc_component(
    name = "mul",
    impl_deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/field25/base:reduce",
        "//sxt/field25/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field25/constant:zero",
        "//sxt/field25/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "neg",
    impl_deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/field25/base:constants",
        "//sxt/field25/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field25/base:constants",
        "//sxt/field25/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "sub",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field25/constant:zero",
    ],
    deps = [
        ":add",
        ":neg",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field25/type:element",
    ],
)
