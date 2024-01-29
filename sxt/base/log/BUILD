load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

cc_library(
    name = "setup",
    srcs = [
        "setup.cc",
    ],
    hdrs = [
        "setup.h",
    ],
    copts = [
        "-stdlib=libc++",
        "-std=c++20",
    ],
    deps = [
        "@com_github_gabime_spdlog//:libspdlog",
    ],
)

cc_library(
    name = "log_impl",
    srcs = [
        "log_impl.cc",
    ],
    hdrs = [
        "log_impl.h",
    ],
    copts = [
        "-stdlib=libc++",
        "-std=c++20",
    ],
    deps = [
        ":setup",
        "@com_github_gabime_spdlog//:libspdlog",
    ],
)

sxt_cc_component(
    name = "log",
    impl_deps = [
        ":log_impl",
    ],
    with_test = False,
)
