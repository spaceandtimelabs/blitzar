load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "abs",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":cneg",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field51/property:sign",
        "//sxt/field51/type:element",
    ],
)

sxt_cc_component(
    name = "add",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field51/type:element",
    ],
)

sxt_cc_component(
    name = "cmov",
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field51/type:element",
    ],
)

sxt_cc_component(
    name = "neg",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":sub",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field51/constant:zero",
        "//sxt/field51/type:element",
    ],
)

sxt_cc_component(
    name = "cneg",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":cmov",
        ":neg",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field51/type:element",
    ],
)

sxt_cc_component(
    name = "sub",
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field51/type:element",
    ],
)

sxt_cc_component(
    name = "mul",
    impl_deps = [
        "//sxt/base/type:int",
        "//sxt/field51/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/field51/type:element",
        "//sxt/field51/type:literal",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "pow22523",
    impl_deps = [
        ":mul",
        ":sq",
        "//sxt/field51/type:element",
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
    name = "sq",
    impl_deps = [
        "//sxt/base/type:int",
        "//sxt/field51/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)
