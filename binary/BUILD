load(
    "//bazel:cuda_dlink.bzl",
    "cuda_dlink",
)

cuda_dlink(
    name = "pedersen_device",
    deps = [
        "//sxt/seqcommit/cbindings:backend",
        "//sxt/seqcommit/cbindings:get_generators",
        "//sxt/seqcommit/cbindings:get_one_commit",
        "//sxt/seqcommit/cbindings:pedersen",
    ],
)

cc_binary(
    name = "libproofs-gpu.so",
    linkopts = [
        "-Wl,--version-script=$(location :libproofs-gpu-export-map.ld)",
        "-static-libstdc++",
        "-static-libgcc",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":libproofs-gpu-export-map.ld",
        ":pedersen_device",
        "//sxt/seqcommit/cbindings:backend",
        "//sxt/seqcommit/cbindings:get_generators",
        "//sxt/seqcommit/cbindings:get_one_commit",
        "//sxt/seqcommit/cbindings:pedersen",
    ],
)
