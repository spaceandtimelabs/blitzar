# load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "cuda_test")
load("@rules_cuda//cuda:defs.bzl", 
     "cuda_library", 
     "cuda_objects", 
)
load(
    "//bazel:cuda_dlink.bzl",
    "cuda_dlink",
)

# We add this -std=c++20 flag, because
# benchmarks could not be compiled without it.
# The `build --cxxopt -std=c++20` flag set in the
# `.bazelrc` file was not passed to the compiler.
# However, this flag is relevant to some modules.
def sxt_copts():
    return [
        "-std=c++20",
        "-Wno-unknown-cuda-version",
        # "-fcoroutines",
    ]

# def sxt_cc_component(
#         name,
#         copts = [],
#         is_cuda = False,
#         with_test = True,
#         alwayslink = False,
#         test_deps = [],
#         deps = [],
#         impl_deps = [],
#         **kwargs):
#     if is_cuda:
#         cuda_library(
#             name = name,
#             hdrs = [
#                 name + ".h",
#             ],
#             srcs = [
#                 name + ".cc",
#             ],
#             copts = sxt_copts() + [
#                 "--device-c",
#                 "-x",
#                 "cuda",
#             ],
#             alwayslink = alwayslink,
#             linkstatic = 1,
#             deps = deps + impl_deps + [
#                 "@local_config_cuda//cuda:cuda_headers",
#                 "@local_config_cuda//cuda:cudart_static",
#             ],
#             visibility = ["//visibility:public"],
#             **kwargs
#         )
#     else:
#         native.cc_library(
#             name = name,
#             hdrs = [
#                 name + ".h",
#             ],
#             srcs = [
#                 name + ".cc",
#             ],
#             copts = sxt_copts() + copts,
#             linkstatic = 1,
#             implementation_deps = impl_deps,
#             deps = deps,
#             alwayslink = alwayslink,
#             visibility = ["//visibility:public"],
#             linkopts = [
#                 "-lm",
#             ],
#             **kwargs
#         )
#     if with_test:
#         deps_p = [
#             ":" + name,
#         ] + deps + test_deps
#         # device_test_name = name + "-device.t"
#         # cuda_dlink(
#         #     name = device_test_name,
#         #     deps = deps_p,
#         # )
#         if is_cuda:
#           pass
#             # cuda_test(
#             #     name = name + ".t",
#             #     srcs = [
#             #         name + ".t.cc",
#             #     ],
#             #     copts = sxt_copts() + copts,
#             #     deps = deps_p + [
#             #         ":" + device_test_name,
#             #     ],
#             #     visibility = ["//visibility:public"],
#             #     **kwargs
#             # )
#         else:
#             native.cc_test(
#                 name = name + ".t",
#                 srcs = [
#                     name + ".t.cc",
#                 ],
#                 copts = sxt_copts() + copts,
#                 deps = deps_p,
#                 visibility = ["//visibility:public"],
#                 **kwargs
#             )

def sxt_cc_component(
        name,
        copts = [],
        is_cuda = False,
        with_test = True,
        alwayslink = False,
        test_deps = [],
        deps = [],
        impl_deps = [],
        **kwargs):
  cuda_objects(
      name = name,
      hdrs = [
          name + ".h",
      ],
      srcs = [
          name + ".cc",
      ],
      copts = sxt_copts(),
      deps = deps + impl_deps + [
          "@local_cuda//:cuda_headers",
          "@local_cuda//:cuda_runtime_static",
      ],
      visibility = ["//visibility:public"],
      **kwargs
  )
  if with_test:
    test_lib = name + ".t-lib"
    deps_p = [
        ":" + name,
    ] + deps + test_deps
    cuda_library(
      name = test_lib,
      copts = sxt_copts(),
      deps = [
        ":" + name,
      ] + test_deps,
      srcs = [
        name + ".t.cc",
      ],
      rdc = True,
    )
    native.cc_test(
        name = name + ".t",
        copts = sxt_copts() + copts,
        deps = [
          ":" + test_lib,
        ] + test_deps + [
          "@com_github_catchorg_catch2//:catch2",
          "@com_github_catchorg_catch2//:catch2_main",
        ],
        visibility = ["//visibility:public"],
        **kwargs
    )
