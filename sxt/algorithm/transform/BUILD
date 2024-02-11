load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "prefix_sum",
    impl_deps = [
      "//sxt/base/device:memory_utility",
      "@local_cuda//:cub",
    ],
    deps = [
      "//sxt/base/container:span",
      "//sxt/base/device:stream",
      "//sxt/memory/management:managed_array",
      "//sxt/memory/resource:async_device_resource",
    ],
    test_deps = [
      "//sxt/memory/management:managed_array",
      "//sxt/memory/resource:managed_device_resource",
      "//sxt/base/device:stream",
      "//sxt/base/device:synchronization",
      "//sxt/base/test:unit_test",
    ],
)
