load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "accumulator",
    with_test = False,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "index_functor",
    with_test = False,
)

sxt_cc_component(
    name = "mapper",
    with_test = False,
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
    ],
    deps = [
        "//sxt/base/error:assert",
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "gather_mapper",
    test_deps = [
        ":mapper",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/error:assert",
        "//sxt/base/macro:cuda_callable",
    ],
)
