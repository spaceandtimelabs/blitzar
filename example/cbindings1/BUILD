cc_library(
    name = "proofs",
    srcs = [
        "//cbindings:libblitzar.so",
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
        "//cbindings:blitzar_api",
    ],
)
