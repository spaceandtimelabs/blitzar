load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "abs",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    with_test = False,
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
    with_test = False,
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
    with_test = False,
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
    with_test = False,
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
    with_test = False,
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
    with_test = False,
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
        "//sxt/base/test:unit_test",
        "//sxt/field51/type:element",
        "//sxt/field51/type:literal",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "invert",
    impl_deps = [
        ":mul",
        ":sq",
        "//sxt/field51/type:element",
    ],
    test_deps = [
        ":mul",
        "//sxt/base/test:unit_test",
        "//sxt/field51/type:element",
        "//sxt/field51/type:literal",
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
    with_test = False,
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
    with_test = False,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "sqmul",
    impl_deps = [
        ":mul",
        ":sq",
    ],
    with_test = False,
)

sxt_cc_component(
    name = "sqrt",
    impl_deps = [
        ":pow22523",
        ":mul",
        ":cmov",
        ":sq",
        ":sub",
        "//sxt/field51/type:element",
        "//sxt/field51/constant:sqrtm1",
        "//sxt/field51/property:zero",
    ],
    test_deps = [
        ":mul",
        "//sxt/base/test:unit_test",
        "//sxt/field51/type:literal",
    ],
)

sxt_cc_component(
    name = "notsquare",
    impl_deps = [
        ":mul",
        ":sq",
        ":sqmul",
        "//sxt/field51/base:byte_conversion",
        "//sxt/field51/type:element",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field51/type:literal",
    ],
)
