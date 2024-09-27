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
  name = "polynomial_utility",
  deps = [
        "//sxt/base/container:span",
  ],
  impl_deps = [
    "//sxt/scalar25/operation:add",
    "//sxt/scalar25/operation:muladd",
    "//sxt/scalar25/type:element",
  ],
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
  impl_deps = [
    ":polynomial_utility",
    "//sxt/base/log:log",
    "//sxt/scalar25/type:element",
    "//sxt/proof/transcript:transcript_utility",
  ],
  test_deps = [
    "//sxt/base/test:unit_test",
  ],
)
