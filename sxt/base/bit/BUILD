load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "iteration",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":count",
        "//sxt/base/error:assert",
    ],
)

sxt_cc_component(
    name = "count",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "load",
    with_test = False,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "permutation",
    deps = [
      ":count",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "span_op",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":count",
        ":iteration",
        "//sxt/base/container:span",
        "//sxt/base/error:assert",
    ],
)

sxt_cc_component(
    name = "store",
    with_test = False,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "zero_equality",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)
