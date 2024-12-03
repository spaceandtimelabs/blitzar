load("//bazel:sxt_build_system.bzl", "sxt_cc_binary")

sxt_cc_binary(
    name = "copy",
    srcs = [
        "copy.m.cc",
    ],
    deps = [
      "//sxt/base/iterator:index_range",
      "//sxt/base/iterator:index_range_utility",
      "//sxt/execution/async:coroutine",
      "//sxt/execution/async:future",
      "//sxt/execution/device:for_each",
      "//sxt/memory/management:managed_array",
    ],
    visibility = ["//visibility:public"],
)
