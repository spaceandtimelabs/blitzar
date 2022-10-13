load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "polymorphic_allocator",
    with_test = False,
)

sxt_cc_component(
    name = "int",
    with_test = False,
)

sxt_cc_component(
    name = "literal",
    with_test = False,
    deps = [
        "//sxt/base/num:hex",
    ],
)

sxt_cc_component(
    name = "remove_cvref",
    with_test = False,
)
