load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "constant",
    with_test = False,
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
    name = "device_cache",
    impl_deps = [
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:state",
        "//sxt/base/device:stream",
        "//sxt/memory/resource:device_resource",
    ],
    test_deps = [
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/device:device_map",
        "//sxt/memory/management:managed_array",
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "mle_utility",
    impl_deps = [
        "//sxt/base/container:span_utility",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:property",
        "//sxt/base/num:divide_up",
        "//sxt/base/num:ceil_log2",
        "//sxt/memory/management:managed_array",
        "//sxt/scalar25/type:element",
    ],
    test_deps = [
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/memory/management:managed_array_fwd",
    ],
)

sxt_cc_component(
    name = "workspace",
    with_test = False,
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
    name = "driver_test",
    impl_deps = [
        ":driver",
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
    impl_deps = [
        ":polynomial_utility",
        "//sxt/base/container:stack_array",
        "//sxt/base/error:panic",
        "//sxt/base/num:ceil_log2",
        "//sxt/execution/async:future",
        "//sxt/memory/management:managed_array",
        "//sxt/scalar25/operation:mul",
        "//sxt/scalar25/operation:sub",
        "//sxt/scalar25/operation:muladd",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
    test_deps = [
        ":driver_test",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":driver",
        ":workspace",
    ],
)

sxt_cc_component(
    name = "transcript_utility",
    impl_deps = [
        "//sxt/scalar25/type:element",
        "//sxt/proof/transcript:transcript_utility",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/proof/transcript",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "chunked_gpu_driver",
    impl_deps = [
        ":device_cache",
        ":fold_gpu",
        ":gpu_driver",
        ":mle_utility",
        ":sum_gpu",
        "//sxt/algorithm/iteration:transform",
        # ":constant",
        # ":partial_polynomial_mapper",
        # ":polynomial_mapper",
        # ":polynomial_utility",
        # "//sxt/algorithm/iteration:for_each",
        # "//sxt/algorithm/reduction:reduction",
        # "//sxt/base/container:stack_array",
        # "//sxt/base/device:stream",
        # "//sxt/base/device:memory_utility",
        "//sxt/base/error:assert",
        "//sxt/base/num:ceil_log2",
        # "//sxt/base/num:constexpr_switch",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        # "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
        # "//sxt/memory/resource:async_device_resource",
        # "//sxt/memory/resource:device_resource",
        # "//sxt/scalar25/operation:mul",
        "//sxt/scalar25/operation:sub",
        # "//sxt/scalar25/operation:muladd",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
    test_deps = [
        ":driver_test",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":driver",
    ],
)

sxt_cc_component(
    name = "gpu_driver",
    impl_deps = [
        ":constant",
        ":partial_polynomial_mapper",
        ":polynomial_mapper",
        ":polynomial_utility",
        "//sxt/algorithm/iteration:for_each",
        "//sxt/algorithm/reduction:reduction",
        "//sxt/base/container:stack_array",
        "//sxt/base/device:stream",
        "//sxt/base/device:memory_utility",
        "//sxt/base/error:panic",
        "//sxt/base/num:ceil_log2",
        "//sxt/base/num:constexpr_switch",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:device_resource",
        "//sxt/scalar25/operation:mul",
        "//sxt/scalar25/operation:sub",
        "//sxt/scalar25/operation:muladd",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
    test_deps = [
        ":driver_test",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":driver",
        "//sxt/memory/management:managed_array_fwd",
    ],
)

sxt_cc_component(
    name = "polynomial_utility",
    impl_deps = [
        "//sxt/scalar25/operation:add",
        "//sxt/scalar25/operation:mul",
        "//sxt/scalar25/operation:sub",
        "//sxt/scalar25/operation:muladd",
        "//sxt/scalar25/operation:neg",
        "//sxt/scalar25/type:element",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "proof_computation",
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
        ":chunked_gpu_driver",
        ":cpu_driver",
        ":gpu_driver",
        ":polynomial_utility",
        ":sumcheck_random",
        ":verification",
        "//sxt/base/container:span_utility",
        "//sxt/base/num:ceil_log2",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/proof/transcript",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/type:element",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/execution/async:future_fwd",
    ],
)

sxt_cc_component(
    name = "fold_gpu",
    impl_deps = [
        ":mle_utility",
        "//sxt/algorithm/iteration:kernel_fit",
        "//sxt/base/error:assert",
        "//sxt/base/device:property",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/num:ceil_log2",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
        "//sxt/scalar25/operation:mul",
        "//sxt/scalar25/operation:muladd",
        "//sxt/scalar25/operation:sub",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/execution/device:for_each",
        "//sxt/execution/device:synchronization",
        "//sxt/execution/kernel:kernel_dims",
        "//sxt/base/iterator:split",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:device_resource",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
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
    name = "sum_gpu",
    impl_deps = [
        ":constant",
        ":device_cache",
        ":mle_utility",
        ":polynomial_mapper2",
        ":reduction_gpu",
        "//sxt/algorithm/reduction:kernel_fit",
        "//sxt/algorithm/reduction:thread_reduction",
        "//sxt/base/device:stream",
        "//sxt/base/device:state",
        "//sxt/base/iterator:split",
        "//sxt/base/num:ceil_log2",
        "//sxt/base/num:constexpr_switch",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/execution/device:for_each",
        #   "//sxt/execution/device:synchronization",
        "//sxt/execution/kernel:kernel_dims",
        #   "//sxt/execution/kernel:launch",
        #   "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/scalar25/operation:add",
        "//sxt/scalar25/operation:mul",
        #   "//sxt/scalar25/operation:accumulator",
        "//sxt/scalar25/type:element",
    ],
    test_deps = [
        ":device_cache",
        # "//sxt/base/device:stream",
        "//sxt/base/iterator:split",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
        # "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/execution/async:future_fwd",
    ],
)

sxt_cc_component(
    name = "reduction_gpu",
    impl_deps = [
        "//sxt/algorithm/base:identity_mapper",
        "//sxt/algorithm/reduction:kernel_fit",
        "//sxt/algorithm/reduction:thread_reduction",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/error:assert",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/execution/device:synchronization",
        "//sxt/execution/kernel:kernel_dims",
        "//sxt/execution/kernel:launch",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/scalar25/operation:add",
        "//sxt/scalar25/operation:accumulator",
        "//sxt/scalar25/type:element",
    ],
    test_deps = [
        ":transcript_utility",
        "//sxt/base/device:property",
        "//sxt/base/device:stream",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/resource:managed_device_resource",
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
    impl_deps = [
        ":polynomial_utility",
        ":transcript_utility",
        "//sxt/base/error:assert",
        "//sxt/base/log:log",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/type:element",
        "//sxt/proof/transcript:transcript_utility",
    ],
    test_deps = [
        ":transcript_utility",
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "partial_polynomial_mapper",
    test_deps = [
        "//sxt/algorithm/base:mapper",
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/operation:overload",
    ],
    deps = [
        ":polynomial_utility",
        "//sxt/base/macro:cuda_callable",
        "//sxt/scalar25/operation:add",
        "//sxt/scalar25/operation:mul",
        "//sxt/scalar25/operation:muladd",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
)

sxt_cc_component(
    name = "polynomial_mapper",
    test_deps = [
        "//sxt/algorithm/base:mapper",
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/operation:overload",
    ],
    deps = [
        ":polynomial_utility",
        "//sxt/base/macro:cuda_callable",
        "//sxt/scalar25/operation:add",
        "//sxt/scalar25/operation:mul",
        "//sxt/scalar25/operation:muladd",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
)

sxt_cc_component(
    name = "polynomial_mapper2",
    test_deps = [
        # "//sxt/algorithm/base:mapper",
        "//sxt/base/test:unit_test",
        # "//sxt/scalar25/operation:overload",
    ],
    deps = [
        ":polynomial_utility",
        "//sxt/base/macro:cuda_callable",
        # "//sxt/scalar25/operation:add",
        # "//sxt/scalar25/operation:mul",
        # "//sxt/scalar25/operation:muladd",
        "//sxt/scalar25/type:element",
        # "//sxt/scalar25/type:literal",
    ],
)
