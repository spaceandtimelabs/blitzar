load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cuda_library(
    name = "exponentiate_gpu",
    srcs = ["exponentiate_gpu.cu"],
    hdrs = ["exponentiate_gpu.h"],
    visibility = ["//visibility:public"],
    copts = [
        '--device-c',
    ],
    deps = [
      "@local_config_cuda//cuda:cuda",
	    "//sxt/base/num:fast_random_number_generator",
	    "//sxt/field51/type:element",
	    "//sxt/curve21/type:element_p3",
	    "//sxt/curve21/operation:scalar_multiply",
    ],
)

cc_library(
    name = "exponentiate_cpu",
    srcs = ["exponentiate_cpu.cc"],
    hdrs = ["exponentiate_cpu.h"],
    visibility = ["//visibility:public"],
    deps = [
	    "//sxt/base/num:fast_random_number_generator",
	    "//sxt/field51/type:element",
	    "//sxt/curve21/type:element_p3",
	    "//sxt/curve21/operation:scalar_multiply",
    ],
)

cc_binary(
    name = "exponentiate",
    srcs = [
        "main.cc",
    ],
    deps = [
      ":exponentiate_cpu",
      ":exponentiate_gpu",
	    "//sxt/curve21/type:element_p3",
    ],
    linkopts = [
        '-x', 'cuda',
        '--cudalog',
    ],
)
