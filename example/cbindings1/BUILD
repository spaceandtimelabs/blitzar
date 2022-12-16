cc_library(
    name = "proofs",
    srcs = [
        "//sxt/cbindings:libproofs-gpu.so",
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
        "//sxt/cbindings:pedersen_capi",
    ],
)
