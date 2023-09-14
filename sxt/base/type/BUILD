load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "int",
    with_test = False,
)

sxt_cc_component(
    name = "literal",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/num:hex",
    ],
)

sxt_cc_component(
    name = "narrow_cast",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "remove_cvref",
    with_test = False,
)

sxt_cc_component(
    name = "raw_stream",
    with_test = False,
)

sxt_cc_component(
    name = "raw_cuda_event",
    with_test = False,
)
