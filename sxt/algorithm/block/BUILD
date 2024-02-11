load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
  name = "runlength_count",
  deps = [
        "@local_cuda//:cub",
  ],
  test_deps = [
    "//sxt/base/test:unit_test",
  ],
)
