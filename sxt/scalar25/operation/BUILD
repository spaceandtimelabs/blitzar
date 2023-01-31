load(
    "//bazel:sxt_component.bzl",
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
        ":mul",
        ":muladd",
        ":product_mapper",
        "//sxt/base/error:assert",
        "//sxt/execution/async:future",
        "//sxt/algorithm/reduction:reduction",
    ],
    is_cuda = True,
    test_deps = [
        ":overload",
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
        "//sxt/scalar25/random:element",
        "//sxt/execution/async:future",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/device:memory_utility",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
    ],
    deps = [
        "//sxt/base/container:span",
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
        "//sxt/scalar25/type:literal",
        "//sxt/scalar25/type:element",
    ],
)

sxt_cc_component(
    name = "product_mapper",
    impl_deps = [
        "//sxt/base/error:assert",
        "//sxt/base/device:memory_utility",
    ],
    test_deps = [
        ":overload",
        "//sxt/execution/base:stream",
        "//sxt/base/test:unit_test",
        "//sxt/scalar25/type:literal",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
    ],
    deps = [
        ":mul",
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/type:raw_stream",
        "//sxt/scalar25/type:element",
    ],
)
