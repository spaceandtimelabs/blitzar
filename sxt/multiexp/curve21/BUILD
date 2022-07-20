load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "multiproduct_cpu_driver",
    impl_deps = [
        "//sxt/curve21/constant:zero",
        "//sxt/base/bit:iteration",
        "//sxt/curve21/operation:add",
        "//sxt/base/container:span_void",
        "//sxt/curve21/type:element_p3",
        "//sxt/multiexp/index:clump2_descriptor",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/index:clump2_marker_utility",
    ],
    test_deps = [
        "//sxt/base/container:span",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/type:element_p3",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/pippenger_multiprod:multiproduct",
        "//sxt/multiexp/index:index_table",
        "//sxt/multiexp/test:generate_curve21_elements",
        "//sxt/multiexp/test:add_curve21_elements",
        "//sxt/multiexp/random:random_multiproduct_descriptor",
        "//sxt/multiexp/random:random_multiproduct_generation",
    ],
    deps = [
        "//sxt/multiexp/pippenger_multiprod:driver",
    ],
)
