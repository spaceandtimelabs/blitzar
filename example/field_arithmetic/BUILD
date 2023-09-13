load("//bazel:sxt_component.bzl", "sxt_cc_binary")

sxt_cc_binary(
    name = "reduce",
    srcs = [
        "reduce.cc",
        "reduce1.cu",
        "reduce1.h",
    ],
    deps = [
        "//sxt/field51/base:reduce",
        "//sxt/field51/operation:add",
        "//sxt/field51/operation:mul",
        "//sxt/field51/type:element",
    ],
)
