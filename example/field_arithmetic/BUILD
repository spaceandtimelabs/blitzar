load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cuda_library(
    name = "reduce1",
    srcs = ["reduce1.cu"],
    hdrs = ["reduce1.h"],
    visibility = ["//visibility:public"],
    copts = [
        "--device-c",
    ],
    deps = [
      "@local_config_cuda//cuda:cuda",
	    "//sxt/field51/type:element",
	    "//sxt/field51/base:reduce",
	    "//sxt/field51/operation:add",
	    "//sxt/field51/operation:mul",
    ],
)

cc_binary(
  name='reduce',
  srcs=['reduce.cc'],
  deps=[
    ':reduce1',
	    "//sxt/field51/type:element",
  ],
  linkopts = [
      '-x', 'cuda'
  ],
)
