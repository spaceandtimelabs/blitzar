load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
  name = "product_kernel",
  deps = [
    "//sxt/base/curve:element",
    "//sxt/base/num:log2p1",
  ],
  test_deps = [
    "//sxt/base/test:unit_test",
  ],
)
