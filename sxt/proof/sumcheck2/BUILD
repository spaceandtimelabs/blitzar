load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "constant",
    with_test = False,
    deps = [
    ],
)

sxt_cc_component(
    name = "workspace",
    with_test = False,
    deps = [
    ],
)

sxt_cc_component(
    name = "device_cache",
    test_deps = [
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/realization:field",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:state",
        "//sxt/base/device:stream",
        "//sxt/base/device:device_map",
        "//sxt/base/field:element",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/management:managed_array",
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
    ":driver_test",
    "//sxt/base/test:unit_test",
  ],
)

sxt_cc_component(
  name = "gpu_driver",
  deps = [
    ":driver",
    # ":polynomial_utility",
    "//sxt/base/error:assert",
    "//sxt/algorithm/iteration:for_each",
    "//sxt/base/device:memory_utility",
    "//sxt/base/device:stream",
    "//sxt/base/num:ceil_log2",
    "//sxt/execution/async:coroutine",
    "//sxt/execution/device:synchronization",
    "//sxt/memory/management:managed_array",
    "//sxt/memory/resource:device_resource",
  ],
  test_deps = [
    ":driver_test",
    "//sxt/base/test:unit_test",
  ],
)

sxt_cc_component(
    name = "polynomial_mapper",
    deps = [
      ":polynomial_utility",
      "//sxt/base/field:element",
      "//sxt/base/macro:cuda_callable",
    ],
    test_deps = [
      "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "polynomial_reducer",
    deps = [
      "//sxt/base/field:element",
      "//sxt/base/macro:cuda_callable",
    ],
    with_test = False,
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
    name = "sum_gpu",
    impl_deps = [
        ":constant",
        # ":device_cache",
        # ":mle_utility",
        ":polynomial_mapper",
        ":polynomial_reducer",
        # ":reduction_gpu",
        "//sxt/algorithm/reduction:kernel_fit",
        "//sxt/algorithm/reduction:thread_reduction",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/device:state",
        "//sxt/base/field:element",
        "//sxt/base/iterator:split",
        "//sxt/base/num:ceil_log2",
        "//sxt/base/num:constexpr_switch",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/execution/device:for_each",
        "//sxt/execution/kernel:kernel_dims",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/resource:async_device_resource",
    ],
    test_deps = [
        # ":device_cache",
        "//sxt/base/iterator:split",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/execution/async:future_fwd",
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
