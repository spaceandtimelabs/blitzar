load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "pollable_event",
    with_test = False,
)

sxt_cc_component(
    name = "test_pollable_event",
    with_test = False,
    deps = [
        ":pollable_event",
    ],
)

sxt_cc_component(
    name = "scheduler",
    impl_deps = [
        ":pollable_event",
    ],
    test_deps = [
        ":test_pollable_event",
        "//sxt/base/test:unit_test",
    ],
)
