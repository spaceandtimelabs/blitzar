load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add",
    impl_deps = [
        "//sxt/field51/type:element",
        "//sxt/field51/operation:add",
        "//sxt/field51/operation:sub",
        "//sxt/field51/operation:mul",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/curve21/constant:zero",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/curve21/type:conversion_utility",
        "//sxt/curve21/type:element_cached",
        "//sxt/curve21/type:element_p1p1",
        "//sxt/curve21/type:element_p2",
        "//sxt/curve21/type:element_p3",
    ],
)

sxt_cc_component(
    name = "cmov",
    impl_deps = [
        "//sxt/curve21/constant:zero",
        "//sxt/curve21/type:element_cached",
        "//sxt/field51/operation:neg",
        "//sxt/field51/operation:cmov",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "double",
    impl_deps = [
        "//sxt/field51/type:element",
        "//sxt/field51/operation:add",
        "//sxt/field51/operation:sub",
        "//sxt/field51/operation:sq",
    ],
    is_cuda = True,
    test_deps = [
        ":add",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/curve21/type:conversion_utility",
        "//sxt/curve21/type:element_p1p1",
        "//sxt/curve21/type:element_p2",
        "//sxt/curve21/type:element_p3",
    ],
)

sxt_cc_component(
    name = "scalar_multiply",
    impl_deps = [
        ":add",
        ":cmov",
        ":double",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/operation:reduce",
        "//sxt/base/container:span",
        "//sxt/curve21/constant:zero",
        "//sxt/curve21/type:conversion_utility",
        "//sxt/curve21/type:element_p1p1",
        "//sxt/curve21/type:element_p2",
        "//sxt/curve21/type:element_p3",
        "//sxt/field51/type:element",
        "//sxt/field51/operation:add",
        "//sxt/field51/operation:sub",
        "//sxt/field51/operation:sq",
    ],
    is_cuda = True,
    test_deps = [
        ":add",
        "//sxt/curve21/type:element_p3",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "overload",
    impl_deps = [
        ":add",
        ":scalar_multiply",
        "//sxt/curve21/type:element_p3",
        "//sxt/scalar25/type:literal",
        "//sxt/scalar25/operation:overload",
    ],
    test_deps = [
        "//sxt/curve21/type:element_p3",
        "//sxt/scalar25/type:literal",
        "//sxt/base/test:unit_test",
    ],
)
