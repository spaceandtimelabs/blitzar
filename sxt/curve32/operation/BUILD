load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add",
    impl_deps = [
        "//sxt/field32/type:element",
        "//sxt/field32/operation:add",
        "//sxt/field32/operation:sub",
        "//sxt/field32/operation:mul",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/curve32/type:conversion_utility",
        "//sxt/curve32/type:element_cached",
        "//sxt/curve32/type:element_p1p1",
        "//sxt/curve32/type:element_p2",
        "//sxt/curve32/type:element_p3",
        "//sxt/field32/constant:d",
    ],
)

sxt_cc_component(
    name = "cmov",
    impl_deps = [
        "//sxt/curve32/constant:identity",
        "//sxt/curve32/type:element_cached",
        "//sxt/field32/operation:neg",
        "//sxt/field32/operation:cmov",
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
    name = "neg",
    impl_deps = [
        ":add",
        "//sxt/curve32/type:element_p3",
        "//sxt/field32/operation:neg",
        "//sxt/field32/operation:cmov",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve32/type:literal",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "double",
    impl_deps = [
        "//sxt/field32/type:element",
        "//sxt/field32/operation:add",
        "//sxt/field32/operation:sub",
        "//sxt/field32/operation:sq",
    ],
    is_cuda = True,
    test_deps = [
        ":add",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/curve32/type:conversion_utility",
        "//sxt/curve32/type:double_impl",
        "//sxt/curve32/type:element_p1p1",
        "//sxt/curve32/type:element_p2",
        "//sxt/curve32/type:element_p3",
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
        "//sxt/curve32/type:conversion_utility",
        "//sxt/curve32/type:element_p1p1",
        "//sxt/curve32/type:element_p2",
        "//sxt/curve32/type:element_p3",
        "//sxt/field32/type:element",
        "//sxt/field32/operation:add",
        "//sxt/field32/operation:sub",
        "//sxt/field32/operation:sq",
    ],
    is_cuda = True,
    test_deps = [
        ":add",
        "//sxt/base/test:unit_test",
        "//sxt/curve32/type:element_p3",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "overload",
    impl_deps = [
        ":add",
        ":neg",
        ":scalar_multiply",
        "//sxt/curve32/type:element_p3",
        "//sxt/scalar25/type:literal",
        "//sxt/scalar25/operation:overload",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve32/type:element_p3",
        "//sxt/scalar25/type:literal",
    ],
)
