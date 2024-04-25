load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add",
    test_deps = [
        "//sxt/base/device:synchronization",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:zero",
        "//sxt/field12/random:element",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/base/field:ptx",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field12/base:constants",
        "//sxt/field12/base:subtract_p",
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "cmov",
    impl_deps = [
        "//sxt/field12/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:one",
        "//sxt/field12/constant:zero",
        "//sxt/field12/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "invert",
    impl_deps = [
        ":pow_vartime",
        "//sxt/field12/property:zero",
        "//sxt/field12/type:element",
    ],
    is_cuda = True,
    test_deps = [
        ":mul",
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:one",
        "//sxt/field12/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "mul",
    impl_deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/field12/base:reduce",
        "//sxt/field12/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:zero",
        "//sxt/field12/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "neg",
    test_deps = [
        "//sxt/base/device:synchronization",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/field12/base:constants",
        "//sxt/field12/constant:zero",
        "//sxt/field12/random:element",
        "//sxt/field12/type:element",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/base/field:ptx",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field12/base:constants",
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "pow_vartime",
    impl_deps = [
        ":mul",
        ":square",
        "//sxt/field12/constant:one",
        "//sxt/field12/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "sqrt",
    impl_deps = [
        ":pow_vartime",
        ":square",
        "//sxt/field12/type:element",
    ],
    is_cuda = True,
    test_deps = [
        ":square",
        "//sxt/base/test:unit_test",
        "//sxt/field12/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "square",
    impl_deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/field12/base:reduce",
        "//sxt/field12/type:element",
    ],
    is_cuda = True,
    test_deps = [
        ":mul",
        "//sxt/base/test:unit_test",
        "//sxt/field12/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "sub",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:zero",
    ],
    deps = [
        ":add",
        ":neg",
        "//sxt/base/macro:cuda_callable",
        "//sxt/field12/type:element",
    ],
)
