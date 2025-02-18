load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "field",
    with_test = False,
    deps = [
      "//sxt/base/field:element",
      "//sxt/scalar25/type:element",
      "//sxt/scalar25/operation:add",
      "//sxt/scalar25/operation:sub",
      "//sxt/scalar25/operation:neg",
      "//sxt/scalar25/operation:mul",
      "//sxt/scalar25/operation:muladd",
    ],
)
