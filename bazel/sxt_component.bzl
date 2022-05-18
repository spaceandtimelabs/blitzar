load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

load(
    "//bazel:cuda_dlink.bzl",
    "cuda_dlink",
)

def sxt_cc_component(
    name, is_cuda=False, with_test=True, alwayslink = False, test_deps=[], deps=[], impl_deps = [], **kwargs):
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
        alwayslink = alwayslink,
        linkstatic = 1,
        # linkopts = [
        #     '-x', 'cuda',
        # ],
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
        linkstatic = 1,
        alwayslink = alwayslink,
        visibility = ["//visibility:public"],
        linkopts = [
            "-lm",
        ],
        **kwargs)
  if with_test:
    deps_p = [
            ":" + name,
    ] + deps + test_deps
    device_test_name = name + '-device.t'
    cuda_dlink(
        name = device_test_name,
        deps = deps_p,
    )
    native.cc_test(
        name = name + ".t",
        srcs = [
            name + ".t.cc",
        ],
        deps = deps_p + [
            ':' + device_test_name,
        ],
        visibility = ["//visibility:public"],
        **kwargs,
    )
