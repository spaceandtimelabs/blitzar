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
        "//sxt/multiexp/index:clump2_descriptor",
        "//sxt/multiexp/index:clump2_marker_utility",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "clump_inputs",
    deps = [
        "//sxt/memory/management:managed_array_fwd",
    ],
    impl_deps = [
        ":driver",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/index:index_table",
        "//sxt/multiexp/index:clump2_descriptor",
        "//sxt/multiexp/index:clump2_descriptor_utility",
        "//sxt/multiexp/index:clump2_marker_utility",
        "//sxt/multiexp/index:marker_transformation",
        "//sxt/multiexp/index:reindex",
    ],
    test_deps = [
        ":test_driver",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/index:index_table",
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
        "//sxt/multiexp/index:partition_marker_utility",
        "//sxt/multiexp/index:marker_transformation",
        "//sxt/multiexp/index:reindex",
    ],
    test_deps = [
        ":test_driver",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/index:index_table",
    ],
)
