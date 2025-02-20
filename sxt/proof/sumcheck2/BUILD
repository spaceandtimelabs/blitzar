load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "workspace",
    with_test = False,
    deps = [
    ],
)

sxt_cc_component(
    name = "driver",
    with_test = False,
    deps = [
      ":workspace",
        "//sxt/execution/async:future_fwd",
      "//sxt/base/container:span",
      "//sxt/base/field:element",
    ],
)

sxt_cc_component(
    name = "driver_test",
    deps = [
        ":driver",
        "//sxt/scalar25/realization:field",
    ],
    impl_deps = [
        ":workspace",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
    with_test = False,
)

sxt_cc_component(
  name = "cpu_driver",
  deps = [
    ":driver",
    ":polynomial_utility",
    "//sxt/base/container:stack_array",
    "//sxt/base/error:assert",
    "//sxt/base/num:ceil_log2",
    "//sxt/execution/async:coroutine",
    "//sxt/memory/management:managed_array",
  ],
  test_deps = [
    "//sxt/base/test:unit_test",
  ],
)

sxt_cc_component(
    name = "polynomial_utility",
    deps = [
      "//sxt/base/container:span",
      "//sxt/base/field:element",
      "//sxt/base/macro:cuda_callable",
    ],
    test_deps = [
      "//sxt/base/test:unit_test",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/realization:field",
        "//sxt/scalar25/type:literal",
    ],
)

sxt_cc_component(
    name = "proof_computation",
    deps = [
      ":sumcheck_transcript",
      ":driver",
      "//sxt/base/container:span",
      "//sxt/base/error:assert",
      "//sxt/base/field:element",
      "//sxt/base/num:ceil_log2",
      "//sxt/execution/async:future",
      "//sxt/execution/async:coroutine",
    ],
    test_deps = [
      "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "sumcheck_transcript",
    with_test = False,
    deps = [
      "//sxt/base/container:span",
      "//sxt/base/field:element",
    ],
)

sxt_cc_component(
    name = "reference_transcript",
    deps = [
      ":sumcheck_transcript",
      "//sxt/base/container:span",
      "//sxt/base/field:element",
      "//sxt/proof/transcript:transcript_utility",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/realization:field",
        "//sxt/scalar25/type:literal",
    ],
)

sxt_cc_component(
    name = "verification",
    # impl_deps = [
    #     ":sumcheck_transcript",
    #     "//sxt/base/error:assert",
    #     "//sxt/base/log:log",
    #     "//sxt/scalar25/type:element",
    # ],
    test_deps = [
        ":reference_transcript",
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/realization:field",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        ":polynomial_utility",
      ":sumcheck_transcript",
        "//sxt/base/container:span",
        "//sxt/base/log:log",
    ],
)
