load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "generator_utility",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:blob_array",
        "//sxt/base/container:span",
        "//sxt/base/error:assert",
    ],
)

sxt_cc_component(
    name = "digit_utility",
    impl_deps = [
        "//sxt/base/bit:count",
        "//sxt/base/num:divide_up",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "exponent_sequence",
    with_test = False,
)

sxt_cc_component(
    name = "exponent_sequence_utility",
    with_test = False,
    deps = [
        ":exponent_sequence",
    ],
)

sxt_cc_component(
    name = "scalar_array",
    impl_deps = [
        "//sxt/base/container:span_utility",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/error:assert",
        "//sxt/execution/async:future",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:generate",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
    ],
    test_deps = [
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/execution/async:future_fwd",
    ],
)
