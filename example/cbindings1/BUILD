cc_library(
    name = "proofs",
    srcs = [
        "//cbindings:libproofs-gpu.so",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "cbind1",
    srcs = [
        "main.cc",
    ],
    deps = [
        ":proofs",
        "//cbindings:proofs_gpu_api",
    ],
)
