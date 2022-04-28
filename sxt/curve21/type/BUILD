load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "conversion_utility",
    deps = [
        ":element_cached",
        ":element_p1p1",
        ":element_p2",
        ":element_p3",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field51/constant:d",
        "//sxt/field51/operation:add",
        "//sxt/field51/operation:mul",
        "//sxt/field51/operation:sub",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "element_cached",
    with_test = False,
    deps = [
        "//sxt/field51/type:element",
    ],
)

sxt_cc_component(
    name = "element_p1p1",
    with_test = False,
    deps = [
        "//sxt/field51/type:element",
    ],
)

sxt_cc_component(
    name = "element_p2",
    with_test = False,
    deps = [
        "//sxt/field51/type:element",
    ],
)

sxt_cc_component(
    name = "element_p3",
    with_test = False,
    deps = [
        "//sxt/field51/type:element",
    ],
    impl_deps = [
        "//sxt/field51/operation:mul",
    ],
)
