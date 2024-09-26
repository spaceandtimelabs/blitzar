load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
  name = "round_sum",
  test_deps = [
    "//sxt/base/test:unit_test",
  ],
)

sxt_cc_component(
  name = "proof_computation",
  deps = [
        "//sxt/base/container:span",
        "//sxt/execution/async:future_fwd",
  ],
  test_deps = [
    "//sxt/base/test:unit_test",
  ],
)

sxt_cc_component(
  name = "verification",
  deps = [
        "//sxt/base/container:span",
  ],
  test_deps = [
    "//sxt/base/test:unit_test",
  ],
)
