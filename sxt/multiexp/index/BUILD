load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "clump2_descriptor",
    with_test = False,
)

sxt_cc_component(
    name = "clump2_descriptor_utility",
    impl_deps = [
        ":clump2_descriptor",
    ],
    test_deps = [
        ":clump2_descriptor",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "clump2_marker_utility",
    impl_deps = [
        ":clump2_descriptor",
    ],
    test_deps = [
        ":clump2_descriptor",
        ":clump2_descriptor_utility",
        ":random_clump2",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "marker_transformation",
    test_deps = [
        ":index_table",
        ":partition_marker_utility",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "partition_marker_utility",
    impl_deps = [
        "//sxt/base/bit:count",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "index_table",
    test_deps = [
        "//sxt/base/test:allocator_aware",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/memory:alloc",
    ],
)

sxt_cc_component(
    name = "index_table_utility",
    impl_deps = [
        ":index_table",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "random_clump2",
    with_test = False,
    deps = [
        ":clump2_descriptor",
    ],
)

sxt_cc_component(
    name = "reindex",
    test_deps = [
        ":index_table",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/functional:function_ref",
    ],
)

sxt_cc_component(
    name = "transpose",
    impl_deps = [
        ":index_table",
        ":index_table_utility",
    ],
    test_deps = [
        ":index_table",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/functional:function_ref",
    ],
)
