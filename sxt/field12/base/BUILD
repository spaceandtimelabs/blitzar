load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "byte_conversion",
    impl_deps = [
        ":montgomery",
        ":reduce",
        "//sxt/base/bit:load",
        "//sxt/base/bit:store",
    ],
    is_cuda = True,
    test_deps = [
        ":constants",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "constants",
    with_test = False,
)

sxt_cc_component(
    name = "montgomery",
    impl_deps = [
        ":constants",
        ":reduce",
        "//sxt/base/field:arithmetic_utility",
    ],
    is_cuda = True,
    test_deps = [
        ":constants",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "reduce",
    is_cuda = True,
    test_deps = [
        ":constants",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":constants",
        ":subtract_p",
        "//sxt/base/field:arithmetic_utility",
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/type:narrow_cast",
    ],
)

sxt_cc_component(
    name = "subtract_p",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":constants",
        "//sxt/base/field:arithmetic_utility",
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/num:cmov",
    ],
)
