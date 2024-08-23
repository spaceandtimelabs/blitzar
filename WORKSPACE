workspace(name = "dev_spaceandtime_blitzar")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "66ffd9315665bfaafc96b52278f57c7e2dd09f5ede279ea6d39b2be471e7e3aa",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
    ],
)

# rules_foreign_cc
git_repository(
    name = "rules_foreign_cc",
    commit = "a87e754",
    remote = "https://github.com/bazelbuild/rules_foreign_cc",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

# libbacktrace
git_repository(
    name = "com_github_ianlancetaylor_libbacktrace",
    build_file_content = """
load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make")

package(default_visibility = ["//visibility:public"])

filegroup(
  name = "all_srcs",
  srcs = glob(
      include = ["**"],
      exclude = ["*.bazel"],
  ),
)

configure_make(
  name = "libbacktrace",
  lib_source = ":all_srcs",
)
  """,
    commit = "14818b7",
    remote = "https://github.com/ianlancetaylor/libbacktrace",
)

# spdlog
git_repository(
    name = "com_github_gabime_spdlog",
    build_file = "//bazel:spdlog.BUILD",
    commit = "5ebfc92",
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
