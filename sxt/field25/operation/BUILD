load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add",
    test_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/field25/constant:zero",
        "//sxt/field25/random:element",
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
    name = "cmov",
    impl_deps = [
        "//sxt/field25/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field25/constant:one",
        "//sxt/field25/constant:zero",
        "//sxt/field25/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "invert",
    impl_deps = [
        ":pow_vartime",
        "//sxt/field25/property:zero",
        "//sxt/field25/type:element",
    ],
    is_cuda = True,
    test_deps = [
        ":mul",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/field25/constant:one",
        "//sxt/field25/random:element",
        "//sxt/field25/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
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
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/field25/base:constants",
        "//sxt/field25/base:montgomery",
        "//sxt/field25/constant:zero",
        "//sxt/field25/random:element",
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
        "//sxt/field25/constant:zero",
        "//sxt/field25/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "pow_vartime",
    impl_deps = [
        ":mul",
        ":square",
        "//sxt/field25/constant:one",
        "//sxt/field25/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field25/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "square",
    impl_deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/field25/base:reduce",
        "//sxt/field25/type:element",
    ],
    is_cuda = True,
    test_deps = [
        ":mul",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/field25/random:element",
        "//sxt/field25/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "sub",
    test_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/field25/constant:zero",
        "//sxt/field25/random:element",
    ],
    deps = [
        ":add",
        ":neg",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field25/type:element",
    ],
)
