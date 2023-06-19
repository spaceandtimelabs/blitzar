load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "awaiter",
    with_test = False,
    deps = [
        ":coroutine_promise",
        ":future",
        "//sxt/base/device:active_device_guard",
        "//sxt/base/device:property",
    ],
)

sxt_cc_component(
    name = "coroutine",
    test_deps = [
        "//sxt/base/device:event",
        "//sxt/base/device:event_utility",
        "//sxt/memory/management:managed_array",
        "//sxt/base/device:active_device_guard",
        "//sxt/base/device:property",
        "//sxt/base/device:state",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":awaiter",
        ":coroutine_promise",
        ":future",
    ],
)

sxt_cc_component(
    name = "coroutine_promise",
    with_test = False,
    deps = [
        ":future",
        ":promise",
        ":task",
        "//sxt/base/error:panic",
    ],
)

sxt_cc_component(
    name = "continuation_fn",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "continuation_fn_utility",
    with_test = False,
)

sxt_cc_component(
    name = "task",
    with_test = False,
)

sxt_cc_component(
    name = "promise",
    with_test = False,
    deps = [
        ":future_state",
        ":promise_future_base",
        ":task",
        "//sxt/base/error:assert",
    ],
)

sxt_cc_component(
    name = "continuation",
    with_test = False,
    deps = [
        ":continuation_fn",
        ":future_state",
        ":future_state_utility",
        ":promise",
        ":task",
    ],
)

sxt_cc_component(
    name = "leaf_continuation",
    with_test = False,
    deps = [
        ":future_state",
        ":task",
    ],
)

sxt_cc_component(
    name = "promise_future_base",
    impl_deps = [
        "//sxt/base/error:assert",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "future_state",
    with_test = False,
    deps = [
        "//sxt/base/error:assert",
    ],
)

sxt_cc_component(
    name = "future_state_utility",
    with_test = False,
    deps = [
        ":continuation_fn",
        ":future_state",
        "//sxt/base/error:assert",
    ],
)

sxt_cc_component(
    name = "future_fwd",
    with_test = False,
)

sxt_cc_component(
    name = "future",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":continuation",
        ":continuation_fn",
        ":continuation_fn_utility",
        ":future_fwd",
        ":future_state",
        ":future_state_utility",
        ":leaf_continuation",
        ":promise",
        ":promise_future_base",
    ],
)

sxt_cc_component(
    name = "future_utility",
    impl_deps = [
        ":coroutine",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":future",
    ],
)
