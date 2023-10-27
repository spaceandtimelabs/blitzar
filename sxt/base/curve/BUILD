load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "element",
    with_test = False,
)

sxt_cc_component(
    name = "example_element",
    deps = [
      "//sxt/base/macro:cuda_callable",
    ],
    test_deps = [
      ":element",
    ]
)
