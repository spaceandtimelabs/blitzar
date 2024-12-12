load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "accumulator",
    test_deps = [
        ":overload",
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    with_test = False,
    deps = [
        ":add",
        ":muladd",
        ":product_mapper",
        "//sxt/base/macro:cuda_callable",
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "add",
    impl_deps = [
        ":reduce",
        "//sxt/base/bit:load",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "inner_product",
    impl_deps = [
        ":accumulator",
        ":add",
        ":mul",
        ":muladd",
        ":product_mapper",
        "//sxt/algorithm/reduction:reduction",
        "//sxt/base/error:assert",
        "//sxt/base/device:property",
        "//sxt/base/device:memory_utility",
        "//sxt/base/iterator:split",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:device_viewable",
        "//sxt/execution/device:for_each",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
    ],
    is_cuda = True,
    test_deps = [
        ":overload",
        "//sxt/base/device:memory_utility",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
        "//sxt/scalar25/random:element",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/device:stream",
        "//sxt/execution/async:future_fwd",
    ],
)

sxt_cc_component(
    name = "neg",
    impl_deps = [
        ":reduce",
        "//sxt/base/bit:load",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "complement",
    impl_deps = [
        ":reduce",
        "//sxt/base/bit:load",
        "//sxt/scalar25/base:reduce",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "sub",
    impl_deps = [
        ":neg",
        ":add",
        "//sxt/base/bit:load",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "mul",
    impl_deps = [
        "//sxt/base/bit:load",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "muladd",
    impl_deps = [
        "//sxt/base/bit:load",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "sq",
    impl_deps = [
        ":mul",
        "//sxt/base/bit:load",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "sqmul",
    impl_deps = [
        ":sq",
        ":mul",
        "//sxt/base/bit:load",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "inv",
    impl_deps = [
        ":sq",
        ":mul",
        ":sqmul",
        "//sxt/base/bit:load",
        "//sxt/scalar25/property:zero",
        "//sxt/scalar25/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "reduce",
    impl_deps = [
        "//sxt/base/bit:load",
        "//sxt/scalar25/base:reduce",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "overload",
    impl_deps = [
        ":add",
        ":mul",
        ":neg",
        ":inv",
        ":sub",
        "//sxt/scalar25/type:element",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:element",
        "//sxt/scalar25/type:literal",
    ],
)

sxt_cc_component(
    name = "product_mapper",
    test_deps = [
        ":overload",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/scalar25/type:literal",
    ],
    deps = [
        ":mul",
        "//sxt/base/macro:cuda_callable",
        "//sxt/scalar25/type:element",
    ],
)
