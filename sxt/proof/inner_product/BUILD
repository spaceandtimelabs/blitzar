load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "workspace",
    with_test = False,
    deps = [
        "//sxt/execution/async:future_fwd",
    ],
)

sxt_cc_component(
    name = "cpu_workspace",
    impl_deps = [
        "//sxt/execution/async:future",
    ],
    with_test = False,
    deps = [
        ":workspace",
        "//sxt/base/container:span",
        "//sxt/execution/async:future_fwd",
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "gpu_workspace",
    impl_deps = [
        "//sxt/base/device:memory_utility",
        "//sxt/execution/async:future",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/resource:pinned_resource",
    ],
    with_test = False,
    deps = [
        ":workspace",
        "//sxt/curve21/type:element_p3",
        "//sxt/memory/management:managed_array",
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "driver",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/execution/async:future_fwd",
    ],
)

sxt_cc_component(
    name = "driver_test",
    with_test = False,
    deps = [
        ":driver",
        ":proof_descriptor",
        ":random_product_generation",
        ":workspace",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:overload",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/ristretto/operation:compression",
        "//sxt/ristretto/operation:overload",
        "//sxt/ristretto/random:element",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/scalar25/operation:inv",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/random:element",
    ],
)

sxt_cc_component(
    name = "generator_fold",
    impl_deps = [
        "//sxt/base/error:assert",
        "//sxt/scalar25/constant:max_bits",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/type:element_p3",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:overload",
        "//sxt/ristretto/operation:compression",
        "//sxt/ristretto/type:literal",
        "//sxt/scalar25/constant:max_bits",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/random:element",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "generator_fold_kernel",
    impl_deps = [
        ":generator_fold",
        "//sxt/algorithm/iteration:for_each",
        "//sxt/base/container:span_utility",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:property",
        "//sxt/base/error:assert",
        "//sxt/base/device:stream",
        "//sxt/base/iterator:index_range",
        "//sxt/base/iterator:index_range_iterator",
        "//sxt/base/iterator:index_range_utility",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:for_each",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/curve21/type:element_p3",
    ],
    is_cuda = True,
    deps = [
        "//sxt/base/container:span",
        "//sxt/execution/async:future_fwd",
    ],
    test_deps = [
        ":generator_fold",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/scalar25/constant:max_bits",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:overload",
        "//sxt/ristretto/operation:compression",
        "//sxt/ristretto/type:literal",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/type:literal",
    ],
)

sxt_cc_component(
    name = "scalar_fold_kernel",
    impl_deps = [
        "//sxt/algorithm/iteration:for_each",
        "//sxt/base/device:memory_utility",
        "//sxt/base/error:assert",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/operation:mul",
        "//sxt/scalar25/operation:muladd",
        "//sxt/execution/async:future",
        "//sxt/execution/async:future_utility",
    ],
    is_cuda = True,
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/execution/async:future_fwd",
    ],
)

sxt_cc_component(
    name = "fold",
    impl_deps = [
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/operation:mul",
        "//sxt/scalar25/operation:muladd",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/random:element",
        "//sxt/scalar25/type:element",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "proof_computation",
    impl_deps = [
        ":driver",
        ":proof_descriptor",
        ":workspace",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/type:element_p3",
        "//sxt/execution/async:future",
        "//sxt/execution/async:coroutine",
        "//sxt/base/num:ceil_log2",
        "//sxt/ristretto/operation:compression",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/proof/transcript:transcript_utility",
        "//sxt/scalar25/type:element",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        ":cpu_driver",
        ":gpu_driver",
        ":proof_descriptor",
        ":random_product_generation",
        "//sxt/base/error:panic",
        "//sxt/base/num:ceil_log2",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:overload",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/proof/transcript",
        "//sxt/ristretto/operation:compression",
        "//sxt/ristretto/type:compressed_element",
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
    name = "proof_descriptor",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/curve21/type:element_p3",
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "random_product_generation",
    impl_deps = [
        ":proof_descriptor",
        "//sxt/base/memory:alloc_utility",
        "//sxt/base/num:ceil_log2",
        "//sxt/scalar25/type:element",
        "//sxt/curve21/type:element_p3",
        "//sxt/ristretto/random:element",
        "//sxt/scalar25/random:element",
        "//sxt/base/error:assert",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/memory:alloc",
    ],
)

sxt_cc_component(
    name = "verification_computation",
    impl_deps = [
        "//sxt/base/error:assert",
        "//sxt/scalar25/operation:mul",
        "//sxt/scalar25/operation:inv",
        "//sxt/scalar25/operation:sq",
        "//sxt/scalar25/operation:neg",
        "//sxt/scalar25/operation:inner_product",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/operation:inv",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "verification_computation_gpu",
    impl_deps = [
        ":verification_computation",
        ":verification_kernel",
        "//sxt/base/container:span_utility",
        "//sxt/base/error:assert",
        "//sxt/scalar25/operation:neg",
        "//sxt/scalar25/operation:inner_product",
        "//sxt/scalar25/type:element",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:pinned_resource",
    ],
    test_deps = [
        ":verification_computation",
        "//sxt/base/device:synchronization",
        "//sxt/base/num:ceil_log2",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/scalar25/operation:inv",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/random:element",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/execution/async:future_fwd",
    ],
)

sxt_cc_component(
    name = "verification_kernel",
    impl_deps = [
        "//sxt/execution/async:future",
        "//sxt/execution/device:synchronization",
        "//sxt/algorithm/iteration:for_each",
        "//sxt/base/device:memory_utility",
        "//sxt/base/error:assert",
        "//sxt/scalar25/operation:mul",
        "//sxt/scalar25/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/type:raw_stream",
        "//sxt/execution/async:future_fwd",
    ],
)

sxt_cc_component(
    name = "cpu_driver",
    impl_deps = [
        ":fold",
        ":generator_fold",
        ":proof_descriptor",
        ":verification_computation",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/type:element_p3",
        "//sxt/execution/async:future",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/curve:multiexponentiation",
        "//sxt/ristretto/operation:compression",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/operation:inner_product",
        "//sxt/scalar25/operation:inv",
        "//sxt/scalar25/constant:max_bits",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        ":driver_test",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":cpu_workspace",
        ":driver",
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "gpu_driver",
    impl_deps = [
        ":cpu_driver",
        ":generator_fold_kernel",
        ":proof_descriptor",
        ":scalar_fold_kernel",
        ":verification_computation_gpu",
        "//sxt/base/container:span_utility",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/error:assert",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/resource:pinned_resource",
        "//sxt/multiexp/base:exponent_sequence_utility",
        "//sxt/multiexp/curve:multiexponentiation",
        "//sxt/scalar25/constant:max_bits",
        "//sxt/scalar25/operation:inner_product",
        "//sxt/scalar25/operation:inv",
        "//sxt/ristretto/operation:compression",
        "//sxt/ristretto/type:compressed_element",
    ],
    test_deps = [
        ":driver_test",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":driver",
        ":gpu_workspace",
        "//sxt/base/container:span",
    ],
)
