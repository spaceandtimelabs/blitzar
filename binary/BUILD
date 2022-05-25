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
    linkopts = [
        "-Wl,--version-script=$(location :libproofs-gpu-export-map.ld)",
        '-static-libstdc++',
        '-static-libgcc',
    ],
    deps = [
        ":pedersen_device",
        ":libproofs-gpu-export-map.ld",
        "//sxt/seqcommit/cbindings:pedersen",
    ],
)
