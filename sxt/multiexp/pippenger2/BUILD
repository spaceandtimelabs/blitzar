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

sxt_cc_component(
  name = "partition_index_kernel",
  deps = [
    "//sxt/base/type:raw_stream",
  ],
  test_deps = [
    "//sxt/base/device:stream",
    "//sxt/base/device:synchronization",
    "//sxt/base/test:unit_test",
    "//sxt/memory/resource:managed_device_resource",
  ],
)

sxt_cc_component(
  name = "partition_index",
  deps = [
    "//sxt/base/container:span",
    "//sxt/base/error:assert",
    "//sxt/execution/async:future_fwd",
  ],
  impl_deps = [
    "//sxt/base/device:stream",
    "//sxt/base/device:memory_utility",
    "//sxt/base/num:divide_up",
    "//sxt/execution/async:future",
  ],
  test_deps = [
    "//sxt/base/test:unit_test",
    "//sxt/memory/resource:managed_device_resource",
  ],
)

sxt_cc_component(
  name = "multiexponentiation",
  deps = [
    "//sxt/base/container:span",
    "//sxt/base/curve:element",
    "//sxt/execution/async:coroutine",
  ],
  test_deps = [
    "//sxt/base/test:unit_test",
  ],
)
