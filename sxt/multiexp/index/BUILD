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
    deps = [
        "//sxt/base/container:span",
    ],
    impl_deps = [
        ":clump2_descriptor",
    ],
    test_deps = [
        ":clump2_descriptor",
        ":clump2_descriptor_utility",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "partition_marker_utility",
    deps = [
        "//sxt/base/container:span",
    ],
    impl_deps = [
        "//sxt/base/bit:count",
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
