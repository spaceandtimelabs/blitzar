load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:zero",
    ],
    deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field12/base:constants",
        "//sxt/field12/base:subtract_p",
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "mul",
    impl_deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/field12/base:reduce",
        "//sxt/field12/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:zero",
        "//sxt/field12/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "neg",
    impl_deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/field12/base:constants",
        "//sxt/field12/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/base:constants",
        "//sxt/field12/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "sub",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:zero",
    ],
    deps = [
        ":add",
        ":neg",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field12/type:element",
    ],
)
