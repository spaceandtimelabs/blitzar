load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "abs",
    with_test = False,
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
    with_test = False,
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
    with_test = False,
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
    with_test = False,
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
    with_test = False,
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
    with_test = False,
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
    name = "invert",
    impl_deps = [
        ":mul",
        ":sq",
        "//sxt/field51/type:element",
    ],
    test_deps = [
        ":mul",
        "//sxt/field51/type:element",
        "//sxt/field51/type:literal",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "pow22523",
    with_test = False,
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
    with_test = False,
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
        "//sxt/field51/type:literal",
        "//sxt/base/test:unit_test",
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
        "//sxt/field51/type:literal",
        "//sxt/base/test:unit_test",
    ],
)
