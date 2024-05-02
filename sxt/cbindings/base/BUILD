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
      "//sxt/base/test:unit_test",
      "//sxt/curve21/type:element_p3",
      "//sxt/curve_g1/type:element_p2",
      "//sxt/curve_bng1/type:element_p2",
    ],
)
