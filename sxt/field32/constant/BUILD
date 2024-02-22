load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "d",
    test_deps = [
      "//sxt/base/test:unit_test",
      "//sxt/field32/type:literal",
    ],
    deps = [
        "//sxt/field32/type:element",
    ],
)

sxt_cc_component(
    name = "invsqrtamd",
    test_deps = [
      "//sxt/base/test:unit_test",
      "//sxt/field32/type:literal",
    ],
    deps = [
        "//sxt/field32/type:element",
    ],
)

sxt_cc_component(
    name = "sqrtm1",
    test_deps = [
      "//sxt/base/test:unit_test",
      "//sxt/field32/type:literal",
    ],
    deps = [
        "//sxt/field32/type:element",
    ],
)

sxt_cc_component(
    name = "zero",
    with_test = False,
    deps = [
        "//sxt/field32/type:element",
    ],
)

sxt_cc_component(
    name = "one",
    with_test = False,
    deps = [
        "//sxt/field32/type:element",
    ],
)
