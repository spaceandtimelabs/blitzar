workspace(name = "dev_spaceandtime_blitzar")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# libbacktrace
git_repository(
    name = "com_github_ianlancetaylor_libbacktrace",
    build_file = "//bazel/libbacktrace:libbacktrace.BUILD",
    commit = "86885d1",
    remote = "https://github.com/ianlancetaylor/libbacktrace",
)

# fmtlib
git_repository(
    name = "com_github_fmtlib_fmt",
    build_file = "//bazel:fmtlib.BUILD",
    commit = "0c9fce2",
    remote = "https://github.com/fmtlib/fmt",
)

# spdlog
git_repository(
    name = "com_github_gabime_spdlog",
    build_file = "//bazel:spdlog.BUILD",
    commit = "a3a0c9d",
    remote = "https://github.com/gabime/spdlog",
)

# catch2
git_repository(
    name = "com_github_catchorg_catch2",
    commit = "53d0d91",
    remote = "https://github.com/catchorg/Catch2",
)

# boost
git_repository(
    name = "com_github_nelhage_rules_boost",
    commit = "ff4fefd",
    # Patch build to add libbacktrace dependency.
    # See https://github.com/nelhage/rules_boost/issues/534
    patches = [
        "//bazel:stacktrace.patch",
    ],
    remote = "https://github.com/nelhage/rules_boost",
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()

# rules_cuda
git_repository(
    name = "rules_cuda",
    commit = "775ba0c",
    remote = "https://github.com/bazel-contrib/rules_cuda",
)

load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")

rules_cuda_dependencies()

register_detected_cuda_toolchains()
