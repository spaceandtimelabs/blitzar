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
        "-std=c++20",
    ],
    implementation_deps = [
        "@com_github_gabime_spdlog//:libspdlog",
    ],
    linkstatic = 1,
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
        "-std=c++20",
    ],
    implementation_deps = [
        ":setup",
        "@com_github_gabime_spdlog//:libspdlog",
    ],
    linkstatic = 1,
)

sxt_cc_component(
    name = "log",
    impl_deps = [
        ":log_impl",
    ],
    with_test = False,
)
