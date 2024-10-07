load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
  name = "transcript_utility",
  impl_deps = [
    "//sxt/scalar25/type:element",
    "//sxt/proof/transcript:transcript_utility",
  ],
  deps = [
        "//sxt/base/container:span",
  ],
  test_deps = [
    "//sxt/base/test:unit_test",
  ],
)

sxt_cc_component(
  name = "driver",
  with_test = False,
  deps = [
    ":workspace",
    "//sxt/base/container:span",
    "//sxt/execution/async:future_fwd",
  ],
)

sxt_cc_component(
  name = "cpu_driver",
  deps = [
    ":driver",
  ],
  impl_deps = [
    ":polynomial_utility",
    "//sxt/base/container:stack_array",
    "//sxt/base/error:panic",
    "//sxt/base/num:ceil_log2",
    "//sxt/execution/async:future",
    "//sxt/memory/management:managed_array",
    "//sxt/scalar25/operation:mul",
    "//sxt/scalar25/operation:muladd",
    "//sxt/scalar25/type:element",
    "//sxt/scalar25/type:literal",
  ],
)

sxt_cc_component(
  name = "workspace",
  with_test = False,
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
    "//sxt/scalar25/operation:mul",
    "//sxt/scalar25/operation:sub",
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
  impl_deps = [
    ":driver",
    ":transcript_utility",
    "//sxt/execution/async:future",
    "//sxt/base/error:assert",
    "//sxt/base/num:ceil_log2",
    "//sxt/execution/async:coroutine",
    "//sxt/scalar25/type:element",
  ],
  test_deps = [
    ":cpu_driver",
    "//sxt/execution/async:future",
    "//sxt/base/test:unit_test",
    "//sxt/proof/transcript:transcript",
    "//sxt/scalar25/operation:overload",
    "//sxt/scalar25/type:element",
  ],
)

sxt_cc_component(
  name = "verification",
  deps = [
        "//sxt/base/container:span",
  ],
  impl_deps = [
    ":polynomial_utility",
    ":transcript_utility",
    "//sxt/base/error:assert",
    "//sxt/base/log:log",
    "//sxt/scalar25/type:element",
    "//sxt/proof/transcript:transcript_utility",
  ],
  test_deps = [
    "//sxt/base/test:unit_test",
    "//sxt/scalar25/type:element",
    "//sxt/scalar25/type:literal",
  ],
)
