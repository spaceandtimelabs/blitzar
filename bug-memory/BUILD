load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
  name = "t_lib",
  srcs = ["t.cc"],
  rdc = True,
)

cc_binary(
  name = "t",
  deps = [
    ":t_lib",
  ]
)
