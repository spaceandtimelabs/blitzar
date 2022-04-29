load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "driver",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/memory/management:managed_array_fwd",
    ],
)

sxt_cc_component(
    name = "test_driver",
    with_test = False,
    deps = [
        ":driver",
    ],
    impl_deps = [
        "//sxt/base/bit:iteration",
        "//sxt/memory/management:managed_array",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "partition_inputs",
    deps = [
        "//sxt/memory/management:managed_array_fwd",
    ],
    impl_deps = [
        ":driver",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/index:index_table",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)
