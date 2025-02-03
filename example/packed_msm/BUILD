load("//bazel:sxt_build_system.bzl", "sxt_cc_binary")


sxt_cc_binary(
    name = "t",
    srcs = [
        "main.cc",
    ],
    deps = [
      "//sxt/multiexp/pippenger2:multiexponentiation_serialization",
        "//sxt/curve_g1/operation:add",
        "//sxt/curve_g1/operation:compression",
        "//sxt/curve_g1/operation:double",
        "//sxt/curve_g1/operation:neg",
        "//sxt/curve_g1/type:compressed_element",
        "//sxt/curve_g1/type:element_p2",
    ],
)
