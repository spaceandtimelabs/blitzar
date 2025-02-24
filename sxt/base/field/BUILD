load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "arithmetic_utility",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/type:int",
        "//sxt/base/type:narrow_cast",
    ],
)

sxt_cc_component(
    name = "element",
    with_test = False,
)

sxt_cc_component(
    name = "accumulator",
    with_test = False,
    deps = [
      ":element",
      "//sxt/base/macro:cuda_callable",
    ]
)
