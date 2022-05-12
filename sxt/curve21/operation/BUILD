load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add",
    is_cuda = True,
    deps = [
        "//sxt/curve21/type:conversion_utility",
        "//sxt/curve21/type:element_cached",
        "//sxt/curve21/type:element_p1p1",
        "//sxt/curve21/type:element_p2",
        "//sxt/curve21/type:element_p3",
        "//sxt/base/macro:cuda_callable",
    ],
    impl_deps = [
        "//sxt/field51/type:element",
        "//sxt/field51/operation:add",
        "//sxt/field51/operation:sub",
        "//sxt/field51/operation:mul",
    ],
    test_deps = [
        "//sxt/curve21/constant:zero",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "cmov",
    is_cuda = True,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
    impl_deps = [
        "//sxt/curve21/constant:zero",
        "//sxt/curve21/type:element_cached",
        "//sxt/field51/operation:neg",
        "//sxt/field51/operation:cmov",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "double",
    is_cuda = True,
    deps = [
        "//sxt/curve21/type:conversion_utility",
        "//sxt/curve21/type:element_p1p1",
        "//sxt/curve21/type:element_p2",
        "//sxt/curve21/type:element_p3",
        "//sxt/base/macro:cuda_callable",
    ],
    impl_deps = [
        "//sxt/field51/type:element",
        "//sxt/field51/operation:add",
        "//sxt/field51/operation:sub",
        "//sxt/field51/operation:square",
    ],
    test_deps = [
        ":add",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "reduce_exponent",
    is_cuda = True,
    impl_deps = [
        "//sxt/base/bit:load",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "scalar_multiply",
    is_cuda = True,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
    impl_deps = [
        ":add",
        ":cmov",
        ":double",
        ":reduce_exponent",
        "//sxt/base/container:span",
        "//sxt/curve21/constant:zero",
        "//sxt/curve21/type:conversion_utility",
        "//sxt/curve21/type:element_p1p1",
        "//sxt/curve21/type:element_p2",
        "//sxt/curve21/type:element_p3",
        "//sxt/field51/type:element",
        "//sxt/field51/operation:add",
        "//sxt/field51/operation:sub",
        "//sxt/field51/operation:square",
    ],
    test_deps = [
        ":add",
        "//sxt/curve21/type:element_p3",
        "//sxt/base/test:unit_test",
    ],
)
