load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "doubling_reduction",
    test_deps = [
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/operation:overload",
        "//sxt/ristretto/type:literal",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/bit:count",
        "//sxt/base/bit:span_op",
        "//sxt/base/container:span",
        "//sxt/base/curve:element",
        "//sxt/base/error:assert",
    ],
)

sxt_cc_component(
    name = "multiexponentiation_cpu_driver",
    test_deps = [
        ":naive_multiproduct_solver",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/type:element_p3",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/pippenger:multiexponentiation",
        "//sxt/multiexp/test:multiexponentiation",
    ],
    deps = [
        ":multiproduct_solver",
        ":multiproducts_combination",
        "//sxt/base/container:blob_array",
        "//sxt/base/curve:element",
        "//sxt/base/error:assert",
        "//sxt/execution/async:future",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/pippenger:driver",
    ],
)

sxt_cc_component(
    name = "multiproducts_combination",
    with_test = False,
    deps = [
        ":doubling_reduction",
        "//sxt/base/bit:span_op",
        "//sxt/base/container:blob_array",
        "//sxt/base/container:span",
        "//sxt/base/container:stack_array",
        "//sxt/base/curve:element",
        "//sxt/base/error:assert",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/management:managed_array_fwd",
        "//sxt/multiexp/base:exponent_sequence",
    ],
)

sxt_cc_component(
    name = "multiproduct_solver",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/curve:element",
        "//sxt/execution/async:future_fwd",
        "//sxt/memory/management:managed_array_fwd",
    ],
)

sxt_cc_component(
    name = "naive_multiproduct_solver",
    with_test = False,
    deps = [
        ":multiproduct_solver",
        "//sxt/base/curve:element",
        "//sxt/execution/async:future",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/base:generator_utility",
        "//sxt/multiexp/index:index_table",
    ],
)
