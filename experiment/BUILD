load("//bazel:sxt_build_system.bzl", "sxt_cc_binary")

sxt_cc_binary(
    name = "shared_memory",
    srcs = [
        "shared_memory.cc",
    ],
    deps = [
      "//sxt/base/device:memory_utility",
      "//sxt/base/device:stream",
      "//sxt/base/device:synchronization",
      "//sxt/memory/management:managed_array",
      "//sxt/memory/resource:device_resource",
    ],
)
