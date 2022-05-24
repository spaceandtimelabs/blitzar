# Note: Adopted from https://github.com/liuliu/rules_cuda

# CUDA Rules for Bazel

This is an up-to-date set of Bazel rules to configure Linux builds with CUDA.

It is directly copied from [Tensorflow](https://github.com/tensorflow/tensorflow/tree/master/third_party/gpus) with some updates. Hence, you need to configure `TF_*` environment variables to make it work.

An example of `.bazelrc` can be found in the root of [this repository](https://github.com/liuliu/rules_cuda/blob/main/.bazelrc). For available configurations, you can find in [`cuda_configure.bzl`](https://github.com/liuliu/rules_cuda/blob/main/gpus/cuda_configure.bzl#L5)

## How to Use

To use rules provided in this repository, you can add following to your `WORKSPACE` file:

```
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

http_archive(
  name = "bazel_skylib",
  sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
  urls = [
    "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
  ],
)

git_repository(
  name = "build_bazel_rules_cuda",
  remote = "https://github.com/liuliu/rules_cuda.git",
  commit = "29f3ced1b7541ae629bbfabe0c07dbfe76f29f4d"
)

load("@build_bazel_rules_cuda//gpus:cuda_configure.bzl", "cuda_configure")

cuda_configure(name = "local_config_cuda")
```

`cuda_library` will be available after `cuda_configure`. An example `BUILD` file could look like this:

```
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cuda_library(
    name = "vectorAdd",
    srcs = ["vectorAdd.cu"],
    hdrs = ["vectorAdd.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cuda",
    ],
)

cc_binary(
  name='main',
  srcs=['main.cc'],
  deps=[':vectorAdd'],
)
```

Note that you need to add `@local_config_cuda//cuda:cuda` explicitly for your target. This is helpful because rather than a sum of `cuda` dependency, you can add specifics if you only use a subset such as `@local_config_cuda//cuda:cudnn`. To see a list of available ones, you can query `bazel query "deps(@local_config_cuda//cuda:cuda)"`

## Updates Differ from TensorFlow

 1. `.cu` is allowed, `gpus/crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc.tpl` modified to allow `.cu` suffix (it is already allowed from Bazel);

 2. Removed `TF_DOWNLOAD_CLANG` option.

## Acknowledgement

This repository incorporated TensorFlow's CUDA rules, as well as some elements from Joe Toth's example repository: [https://github.com/joetoth/bazel_cuda](https://github.com/joetoth/bazel_cuda). TensorFlow rules updated from commit bde54d7aeaa23a47e1c5900464a8871d643e2e8e.
