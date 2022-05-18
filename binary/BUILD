load(
    "//bazel:cuda_dlink.bzl",
    "cuda_dlink",
)


cuda_dlink(
    name = "pedersen_device",
    deps = [
        "//sxt/seqcommit/cbindings:pedersen",
    ],
)

cc_binary(
    name = "libproofs-gpu.so",
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":pedersen_device",
        "//sxt/seqcommit/cbindings:pedersen",
    ],
)
