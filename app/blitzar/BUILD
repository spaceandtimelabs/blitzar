load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_binary",
)

sxt_cc_binary(
    name = "blitzar",
    srcs = [
        "main.cc",
    ],
    deps = [
        "//sxt/base/error:panic",
        "//sxt/base/num:divide_up",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/type:element_p3",
        "//sxt/multiexp/pippenger2:partition_table",
        "//sxt/ristretto/random:element",
    ],
)
