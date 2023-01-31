load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "mapper",
    with_test = False,
    deps = [
        "//sxt/base/type:raw_stream",
    ],
)

sxt_cc_component(
    name = "reducer",
    with_test = False,
)

sxt_cc_component(
    name = "reducer_utility",
    with_test = False,
    deps = [
        "//sxt/algorithm/base:mapper",
        "//sxt/algorithm/base:reducer",
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "identity_mapper",
    test_deps = [
        ":mapper",
        "//sxt/base/test:unit_test",
        "//sxt/execution/base:stream",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
        "//sxt/base/device:synchronization",
    ],
    deps = [
        "//sxt/base/device:memory_utility",
        "//sxt/base/error:assert",
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/type:raw_stream",
    ],
)
