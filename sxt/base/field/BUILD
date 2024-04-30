load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "arithmetic_utility",
    test_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/type:int",
        "//sxt/base/type:narrow_cast",
    ],
)

sxt_cc_component(
    name = "ptx",
    with_test = False,
    deps = [
        "//sxt/base/num:cmov",
    ],
)
