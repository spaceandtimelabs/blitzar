load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
  name = "product_kernel",
  deps = [
    "//sxt/algorithm/iteration:for_each",
    "//sxt/base/curve:element",
    "//sxt/base/macro:cuda_callable",
    "//sxt/base/num:log2p1",
  ],
  test_deps = [
    "//sxt/base/curve:example_element",
    "//sxt/base/device:synchronization",
    "//sxt/base/device:stream",
    "//sxt/base/test:unit_test",
    "//sxt/memory/resource:managed_device_resource",
  ],
)
