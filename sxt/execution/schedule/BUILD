load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "pollable_event",
    with_test = False,
)

sxt_cc_component(
    name = "pending_event",
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
    name = "test_pending_event",
    with_test = False,
    deps = [
        ":pending_event",
    ],
)

sxt_cc_component(
    name = "active_scheduler",
    impl_deps = [
        ":pollable_event",
    ],
    test_deps = [
        ":test_pollable_event",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/functional:function_ref",
    ],
)

sxt_cc_component(
    name = "scheduler",
    impl_deps = [
        ":pollable_event",
        "//sxt/base/device:property",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        ":test_pending_event",
        ":test_pollable_event",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":active_scheduler",
        ":pending_scheduler",
    ],
)

sxt_cc_component(
    name = "pending_scheduler",
    impl_deps = [
        "//sxt/base/error:assert",
    ],
    test_deps = [
        ":test_pending_event",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":pending_event",
    ],
)
