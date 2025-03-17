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
        "//sxt/base/device:device_map",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:state",
        "//sxt/base/device:stream",
        "//sxt/base/field:element",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
    ],
)

sxt_cc_component(
    name = "driver",
    with_test = False,
    deps = [
        ":workspace",
        "//sxt/base/container:span",
        "//sxt/base/field:element",
        "//sxt/execution/async:future_fwd",
    ],
)

sxt_cc_component(
    name = "driver_test",
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
    deps = [
        ":driver",
        "//sxt/scalar25/realization:field",
    ],
)

sxt_cc_component(
    name = "cpu_driver",
    test_deps = [
        ":driver_test",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":driver",
        ":polynomial_utility",
        "//sxt/base/container:stack_array",
        "//sxt/base/error:assert",
        "//sxt/base/num:ceil_log2",
        "//sxt/execution/async:coroutine",
        "//sxt/memory/management:managed_array",
    ],
)

sxt_cc_component(
    name = "fold_gpu",
    test_deps = [
        "//sxt/base/iterator:split",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/realization:field",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        ":mle_utility",
        "//sxt/algorithm/iteration:kernel_fit",
        "//sxt/base/container:span",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:property",
        "//sxt/base/device:stream",
        "//sxt/base/error:assert",
        "//sxt/base/field:element",
        "//sxt/base/iterator:split",
        "//sxt/base/num:ceil_log2",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/execution/device:for_each",
        "//sxt/execution/device:synchronization",
        "//sxt/execution/kernel:kernel_dims",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:device_resource",
        "//sxt/scalar25/operation:mul",
        "//sxt/scalar25/operation:muladd",
        "//sxt/scalar25/operation:sub",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
)

sxt_cc_component(
    name = "gpu_driver",
    test_deps = [
        ":driver_test",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":driver",
        ":sum_gpu",
        "//sxt/algorithm/iteration:for_each",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/error:assert",
        "//sxt/base/num:ceil_log2",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
    ],
)

sxt_cc_component(
    name = "chunked_gpu_driver",
    test_deps = [
        ":driver_test",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":device_cache",
        ":driver",
        ":fold_gpu",
        ":gpu_driver",
        ":mle_utility",
        ":sum_gpu",
        "//sxt/algorithm/iteration:transform",
        "//sxt/base/error:assert",
        "//sxt/base/num:ceil_log2",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/memory/management:managed_array",
    ],
)

sxt_cc_component(
    name = "mle_utility",
    impl_deps = [
        "//sxt/base/container:span_utility",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:property",
        "//sxt/base/container:span",
        "//sxt/base/field:element",
        "//sxt/base/num:divide_up",
        "//sxt/base/num:ceil_log2",
        "//sxt/memory/management:managed_array",
    ],
    test_deps = [
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/scalar25/realization:field",
        "//sxt/scalar25/type:literal",
    ],
)

sxt_cc_component(
    name = "polynomial_mapper",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/realization:field",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        ":polynomial_utility",
        "//sxt/base/field:element",
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "polynomial_reducer",
    with_test = False,
    deps = [
        "//sxt/base/field:element",
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "polynomial_utility",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/realization:field",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/field:element",
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "proof_computation",
    test_deps = [
        ":chunked_gpu_driver",
        ":cpu_driver",
        ":gpu_driver",
        ":mle_utility",
        ":reference_transcript",
        ":sumcheck_random",
        ":verification",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/fieldgk/realization:field",
        "//sxt/fieldgk/type:literal",
        "//sxt/proof/transcript",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/realization:field",
    ],
    deps = [
        ":driver",
        ":sumcheck_transcript",
        "//sxt/base/container:span",
        "//sxt/base/error:assert",
        "//sxt/base/field:element",
        "//sxt/base/num:ceil_log2",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
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
    name = "sumcheck_random",
    impl_deps = [
        "//sxt/base/error:assert",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/scalar25/random:element",
        "//sxt/scalar25/type:element",
    ],
    with_test = False,
    deps = [
        ":constant",
    ],
)

sxt_cc_component(
    name = "reference_transcript",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/realization:field",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        ":sumcheck_transcript",
        "//sxt/base/container:span",
        "//sxt/base/field:element",
        "//sxt/proof/transcript:transcript_utility",
    ],
)

sxt_cc_component(
    name = "reduction_gpu",
    test_deps = [
        "//sxt/base/device:property",
        "//sxt/base/device:stream",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/realization:field",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/algorithm/base:identity_mapper",
        "//sxt/algorithm/reduction:kernel_fit",
        "//sxt/algorithm/reduction:thread_reduction",
        "//sxt/base/container:span",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/error:assert",
        "//sxt/base/field:accumulator",
        "//sxt/base/field:element",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/execution/async:future_fwd",
        "//sxt/execution/device:synchronization",
        "//sxt/execution/kernel:kernel_dims",
        "//sxt/execution/kernel:launch",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
    ],
)

sxt_cc_component(
    name = "sum_gpu",
    impl_deps = [
        ":constant",
        ":device_cache",
        ":mle_utility",
        ":polynomial_mapper",
        ":polynomial_reducer",
        ":reduction_gpu",
        "//sxt/algorithm/reduction:kernel_fit",
        "//sxt/algorithm/reduction:thread_reduction",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:split",
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
        "//sxt/scalar25/realization:field",
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
        "//sxt/base/log",
    ],
)
