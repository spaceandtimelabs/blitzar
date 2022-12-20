load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add",
    impl_deps = [
        ":reduce",
        "//sxt/base/bit:load",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "neg",
    impl_deps = [
        ":reduce",
        "//sxt/base/bit:load",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "complement",
    impl_deps = [
        ":reduce",
        "//sxt/base/bit:load",
        "//sxt/scalar25/base:reduce",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "sub",
    impl_deps = [
        ":neg",
        ":add",
        "//sxt/base/bit:load",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "mul",
    impl_deps = [
        "//sxt/base/bit:load",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "muladd",
    impl_deps = [
        "//sxt/base/bit:load",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "sq",
    impl_deps = [
        ":mul",
        "//sxt/base/bit:load",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "sqmul",
    impl_deps = [
        ":sq",
        ":mul",
        "//sxt/base/bit:load",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "inv",
    impl_deps = [
        ":sq",
        ":mul",
        ":sqmul",
        "//sxt/base/bit:load",
        "//sxt/scalar25/property:zero",
        "//sxt/scalar25/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "reduce",
    impl_deps = [
        "//sxt/base/bit:load",
        "//sxt/scalar25/base:reduce",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)
