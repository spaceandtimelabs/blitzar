load("//bazel:sxt_build_system.bzl", "sxt_cc_binary")

sxt_cc_binary(
    name = "copy",
    srcs = [
        "copy.m.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//sxt/algorithm/iteration:for_each",
        "//sxt/base/container:span",
        "//sxt/base/container:span_utility",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/iterator:index_range",
        "//sxt/base/iterator:index_range_utility",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/execution/device:copy",
        "//sxt/execution/device:for_each",
        "//sxt/execution/device:synchronization",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:pinned_resource",
    ],
)
