cc_library(
    name = "proofs",
    srcs = [
        "//binary:libproofs-gpu.so",
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
        "//sxt/seqcommit/cbindings:pedersen_capi",
    ],
)
