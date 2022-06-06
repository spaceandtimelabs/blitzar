load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "abs",
    deps = [
        ":cneg",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field51/property:sign",
        "//sxt/field51/type:element",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "add",
    deps = [
        "//sxt/field51/type:element",
        "//sxt/base/macro:cuda_callable",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "cmov",
    is_cuda = True,
    deps = [
        "//sxt/field51/type:element",
        "//sxt/base/macro:cuda_callable",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "neg",
    deps = [
        ":sub",
        "//sxt/field51/constant:zero",
        "//sxt/field51/type:element",
        "//sxt/base/macro:cuda_callable",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "cneg",
    deps = [
        ":cmov",
        ":neg",
        "//sxt/field51/type:element",
        "//sxt/base/macro:cuda_callable",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "sub",
    is_cuda = True,
    deps = [
        "//sxt/field51/type:element",
        "//sxt/base/macro:cuda_callable",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "mul",
    is_cuda = True,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
    impl_deps = [
        "//sxt/base/type:int",
        "//sxt/field51/type:element",
    ],
    test_deps = [
        "//sxt/field51/type:element",
        "//sxt/field51/type:literal",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "pow22523",
    is_cuda = True,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
    impl_deps = [
        ":mul",
        ":square",
        "//sxt/field51/type:element",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "square",
    is_cuda = True,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
    impl_deps = [
        "//sxt/base/type:int",
        "//sxt/field51/type:element",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)
