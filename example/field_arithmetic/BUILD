load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "reduce1",
    srcs = ["reduce1.cu"],
    hdrs = ["reduce1.h"],
    copts = [],
    visibility = ["//visibility:public"],
    deps = [
        "//sxt/field51/base:reduce",
        "//sxt/field51/operation:add",
        "//sxt/field51/operation:mul",
        "//sxt/field51/type:element",
        "@local_cuda//:cuda",
    ],
)

cc_binary(
    name = "reduce",
    srcs = ["reduce.cc"],
    linkopts = [
        "-x",
        "cuda",
    ],
    deps = [
        ":reduce1",
        "//sxt/field51/type:element",
    ],
)
