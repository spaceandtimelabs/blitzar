load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add",
    impl_deps = [
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:zero",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field12/base:arithmetic_utility",
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "cmov",
    impl_deps = [
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:one",
        "//sxt/field12/constant:zero",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "invert",
    impl_deps = [
        ":pow_vartime",
        "//sxt/field12/property:zero",
    ],
    test_deps = [
        ":mul",
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:one",
    ],
    deps = [
        ":neg",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "mul",
    impl_deps = [
        "//sxt/field12/base:arithmetic_utility",
        "//sxt/field12/base:reduce",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:zero",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "neg",
    impl_deps = [
        "//sxt/field12/base:arithmetic_utility",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field12/base:constants",
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "pow_vartime",
    impl_deps = [
        ":mul",
        ":square",
        "//sxt/field12/constant:one",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "sqrt",
    impl_deps = [
        ":pow_vartime",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":neg",
        ":square",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "square",
    impl_deps = [
        "//sxt/field12/base:arithmetic_utility",
        "//sxt/field12/base:reduce",
    ],
    test_deps = [
        ":mul",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field12/type:element",
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
