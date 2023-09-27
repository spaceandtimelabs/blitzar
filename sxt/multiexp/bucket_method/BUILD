load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "accumulate_kernel",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/base/device:synchronization",
    ],
    deps = [
        "//sxt/base/curve:element",
        "//sxt/base/num:divide_up",
    ],
)

sxt_cc_component(
    name = "combination_kernel",
    test_deps = [
        "//sxt/algorithm/base:identity_mapper",
        "//sxt/algorithm/reduction:test_reducer",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/base/device:synchronization",
    ],
    deps = [
        "//sxt/algorithm/base:reducer",
        "//sxt/base/num:divide_up",
        "//sxt/base/curve:element",
    ],
)

sxt_cc_component(
    name = "bucket_accumulation",
    deps = [
      ":accumulate_kernel",
      ":combination_kernel",
      "//sxt/base/error:assert",
        "//sxt/base/container:span",
        "//sxt/base/curve:element",
        "//sxt/base/device:stream",
        "//sxt/base/iterator:index_range",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/execution/device:for_each",
        "//sxt/execution/device:device_viewable",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:managed_device_resource",
    ],
)
