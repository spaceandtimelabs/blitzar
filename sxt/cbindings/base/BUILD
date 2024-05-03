load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "curve_id",
    with_test = False,
    deps = [
    ],
)

sxt_cc_component(
    name = "curve_id_utility",
    deps = [
      ":curve_id",
      "//sxt/base/error:panic",
      "//sxt/base/type:type",
      "//sxt/base/test:unit_test",
      "//sxt/curve_bng1/operation:add",
      "//sxt/curve_bng1/operation:double",
      "//sxt/curve_bng1/operation:neg",
      "//sxt/curve_bng1/type:element_p2",
      "//sxt/curve_g1/operation:add",
      "//sxt/curve_g1/operation:double",
      "//sxt/curve_g1/operation:neg",
      "//sxt/curve_g1/type:element_p2",
      "//sxt/curve21/operation:add",
      "//sxt/curve21/operation:double",
      "//sxt/curve21/operation:neg",
      "//sxt/curve21/type:element_p3",
    ],
)

sxt_cc_component(
    name = "multiexp_handle",
    with_test = False,
    deps = [
      ":curve_id",
      "//sxt/multiexp/pippenger2:partition_table_accessor_base",
    ],
)
