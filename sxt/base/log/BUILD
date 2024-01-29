load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

cc_library(
  name = "setup",
  copts = [
    "-stdlib=libc++",
    "-std=c++20",
  ],
  hdrs = [
    "setup.h",
  ],
  srcs = [
    "setup.cc",
  ],
  deps = [
    "@com_github_gabime_spdlog//:libspdlog",
  ],
)

cc_library(
  name = "log_impl",
  copts = [
    "-stdlib=libc++",
    "-std=c++20",
  ],
  hdrs = [
    "log_impl.h",
  ],
  srcs = [
    "log_impl.cc",
  ],
  deps = [
    ":setup",
    "@com_github_gabime_spdlog//:libspdlog",
  ],
)

sxt_cc_component(
  name = "log",
  with_test = False,
  impl_deps = [
    ":log_impl",
  ],
)
