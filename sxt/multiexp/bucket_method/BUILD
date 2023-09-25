load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "accumulate_kernel",
    test_deps = [
        "//sxt/algorithm/base:identity_mapper",
        "//sxt/algorithm/reduction:test_reducer",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/base/device:synchronization",
    ],
    deps = [
        "//sxt/algorithm/base:mapper",
        "//sxt/algorithm/base:reducer",
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
        "//sxt/base/container:span",
        "//sxt/base/curve:element",
    ],
)
