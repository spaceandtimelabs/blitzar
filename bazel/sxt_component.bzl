load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

def sxt_cc_component(
    name, is_cuda=False, with_test=True, test_deps=[], deps=[], impl_deps = [], **kwargs):
  if is_cuda:
    cuda_library(
        name = name,
        hdrs = [
            name + ".h",
        ],
        srcs = [
            name + ".cc",
        ],
        copts = [
            '--device-c',
            '-x',
            'cuda',
        ],
        linkstatic = 1,
        linkopts = [
            '-x',
            'cuda',
        ],
        deps = deps + impl_deps + [
          "@local_config_cuda//cuda:cuda",
        ],
        visibility = ["//visibility:public"],
        **kwargs)
  else:
    native.cc_library(
        name = name,
        hdrs = [
            name + ".h",
        ],
        srcs = [
            name + ".cc",
        ],
        deps = deps + impl_deps,
        visibility = ["//visibility:public"],
        **kwargs)
  if with_test:
    native.cc_test(
        name = name + ".t",
        srcs = [
            name + ".t.cc",
        ],
        deps = [
            ":" + name,
        ] + deps + test_deps,
        visibility = ["//visibility:public"],
        **kwargs,
    )
