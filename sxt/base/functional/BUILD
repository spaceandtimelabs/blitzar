load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "function_ref",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "move_only_function_handle",
    with_test = False,
)

sxt_cc_component(
    name = "move_only_function_handle_impl",
    with_test = False,
    deps = [
        ":move_only_function_handle",
    ],
)

sxt_cc_component(
    name = "move_only_function",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":move_only_function_handle",
        ":move_only_function_handle_impl",
        "//sxt/base/error:assert",
    ],
)
