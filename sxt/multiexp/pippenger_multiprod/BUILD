load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "active_count",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "active_offset",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "driver",
    impl_deps = [
        "//sxt/base/container:span_void",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "test_driver",
    impl_deps = [
        "//sxt/base/bit:iteration",
        "//sxt/base/container:span_void",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/index:clump2_descriptor",
        "//sxt/multiexp/index:clump2_marker_utility",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    with_test = False,
    deps = [
        ":driver",
    ],
)

sxt_cc_component(
    name = "clump_inputs",
    impl_deps = [
        ":active_offset",
        ":driver",
        ":prune",
        ":reduction_stats",
        "//sxt/base/container:span_void",
        "//sxt/multiexp/index:clump2_descriptor",
        "//sxt/multiexp/index:clump2_descriptor_utility",
        "//sxt/multiexp/index:clump2_marker_utility",
        "//sxt/multiexp/index:marker_transformation",
        "//sxt/multiexp/index:reindex",
    ],
    test_deps = [
        ":reduction_stats",
        ":test_driver",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/index:index_table",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "clump_outputs",
    impl_deps = [
        ":active_offset",
        "//sxt/multiexp/index:clump2_descriptor",
        "//sxt/multiexp/index:clump2_descriptor_utility",
        "//sxt/multiexp/index:clump2_marker_utility",
        "//sxt/multiexp/index:index_table",
        "//sxt/multiexp/index:index_table_utility",
        "//sxt/multiexp/index:marker_transformation",
        "//sxt/multiexp/index:reindex",
        "//sxt/multiexp/index:transpose",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/multiexp/index:index_table",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "multiproduct_params",
    with_test = False,
)

sxt_cc_component(
    name = "multiproduct_params_computation",
    impl_deps = [
        ":multiproduct_params",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "multiproduct",
    impl_deps = [
        ":clump_inputs",
        ":clump_outputs",
        ":driver",
        ":multiproduct_params",
        ":multiproduct_params_computation",
        ":partition_inputs",
        ":product_table_normalization",
        ":prune",
        ":reduction_stats",
        "//sxt/base/container:span_void",
        "//sxt/base/iterator:counting_iterator",
        "//sxt/multiexp/index:index_table",
    ],
    test_deps = [
        ":multiproduct_params",
        ":test_driver",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/index:index_table",
        "//sxt/multiexp/random:random_multiproduct_descriptor",
        "//sxt/multiexp/random:random_multiproduct_generation",
        "//sxt/multiexp/test:add_ints",
        "//sxt/multiexp/test:int_generation",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "partition_inputs",
    impl_deps = [
        ":active_offset",
        ":driver",
        ":prune",
        ":reduction_stats",
        "//sxt/base/container:span_void",
        "//sxt/multiexp/index:index_table",
        "//sxt/multiexp/index:partition_marker_utility",
        "//sxt/multiexp/index:marker_transformation",
        "//sxt/multiexp/index:reindex",
    ],
    test_deps = [
        ":reduction_stats",
        ":test_driver",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/index:index_table",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "product_table_normalization",
    impl_deps = [
        "//sxt/multiexp/index:index_table",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/multiexp/index:index_table",
    ],
)

sxt_cc_component(
    name = "prune",
    impl_deps = [
        ":active_count",
        "//sxt/base/iterator:counting_iterator",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/multiexp/index:index_table",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "reduction_stats",
    with_test = False,
)
