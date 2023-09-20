load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "accumulate_kernel",
    with_test = False,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
      "//sxt/algorithm/base:mapper",
      "//sxt/algorithm/base:reducer",
      "//sxt/base/num:divide_up",
    ],
)
