load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "abs",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field32/type:literal",
    ],
    deps = [
        ":cneg",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field32/property:sign",
        "//sxt/field32/type:element",
    ],
)

sxt_cc_component(
    name = "add",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field32/type:literal",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field32/type:element",
    ],
)

sxt_cc_component(
    name = "cmov",
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field32/constant:one",
        "//sxt/field32/constant:zero",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field32/type:element",
    ],
)

sxt_cc_component(
    name = "neg",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field32/base:byte_conversion",
        "//sxt/field32/type:literal",
    ],
    deps = [
        ":sub",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field32/constant:zero",
        "//sxt/field32/type:element",
    ],
)

sxt_cc_component(
    name = "cneg",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field32/type:literal",
    ],
    deps = [
        ":cmov",
        ":neg",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field32/type:element",
    ],
)

sxt_cc_component(
    name = "sub",
    impl_deps = [
        "//sxt/field32/base:reduce",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field32/type:literal",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/field32/type:element",
    ],
)

sxt_cc_component(
    name = "mul",
    impl_deps = [
        "//sxt/base/type:int",
        "//sxt/field32/base:reduce",
        "//sxt/field32/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field32/type:element",
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
        "//sxt/field32/type:element",
    ],
    test_deps = [
        ":mul",
        "//sxt/base/test:unit_test",
        "//sxt/field32/type:element",
        "//sxt/field32/type:literal",
    ],
)

sxt_cc_component(
    name = "pow22523",
    impl_deps = [
        ":mul",
        ":sq",
        "//sxt/field32/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field32/type:element",
        "//sxt/field32/type:literal",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "sq",
    impl_deps = [
        "//sxt/base/type:int",
        "//sxt/field32/base:reduce",
        "//sxt/field32/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field32/base:byte_conversion",
        "//sxt/field32/type:element",
        "//sxt/field32/type:literal",
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
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field32/type:element",
        "//sxt/field32/type:literal",
    ],
)

sxt_cc_component(
    name = "sqrt",
    impl_deps = [
        ":pow22523",
        ":mul",
        ":cmov",
        ":sq",
        ":sub",
        "//sxt/field32/type:element",
        "//sxt/field32/constant:sqrtm1",
        "//sxt/field32/property:zero",
    ],
    test_deps = [
        ":mul",
        "//sxt/base/test:unit_test",
        "//sxt/field32/type:literal",
    ],
)

sxt_cc_component(
    name = "notsquare",
    impl_deps = [
        ":mul",
        ":sq",
        ":sqmul",
        "//sxt/field32/base:byte_conversion",
        "//sxt/field32/type:element",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field32/type:literal",
    ],
)
