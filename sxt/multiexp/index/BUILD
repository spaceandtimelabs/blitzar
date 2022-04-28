load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "partition_marker_utility",
    deps = [
        "//sxt/base/bit:count",
        "//sxt/base/container:span",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "partition",
    deps = [
        "//sxt/base/container:span",
    ],
    impl_deps = [
        ":partition_marker_utility",
    ],
    test_deps = [
        ":index_table",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "index_table",
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/type:polymorphic_allocator",
    ],
    test_deps = [
        "//sxt/base/test:allocator_aware",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "reindex",
    deps = [
        "//sxt/base/container:span",
    ],
    test_deps = [
        ":index_table",
        "//sxt/base/test:unit_test",
    ],
)
