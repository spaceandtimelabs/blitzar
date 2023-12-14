load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add",
    test_deps = [
        ":example_element",
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:zero",
        "//sxt/field12/type:element",
    ],
    deps = [
        ":arithmetic_utility",
        ":element",
        ":subtract_p",
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "arithmetic_utility",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/type:int",
        "//sxt/base/type:narrow_cast",
    ],
)

sxt_cc_component(
    name = "element",
    with_test = False,
)

sxt_cc_component(
    name = "example_element",
    test_deps = [
        ":element",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "subtract_p",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":arithmetic_utility",
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/num:cmov",
    ],
)
