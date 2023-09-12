workspace(name = "dev_spaceandtime_blitzar")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

http_archive(
    name = "bazel_skylib",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    urls = [
        "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
    ],
)

local_repository(
    name = "build_bazel_rules_cuda",
    path = "third_party/rules_cuda",
)

load("@build_bazel_rules_cuda//gpus:cuda_configure.bzl", "cuda_configure")

cuda_configure(name = "local_config_cuda")

git_repository(
    name = "com_github_catchorg_catch2",
    commit = "6f21a3609cea360846a0ca93be55877cca14c86d",
    remote = "https://github.com/catchorg/Catch2",
)

git_repository(
    name = "rules_cuda",
    commit = "22a46e6",
    remote = "https://github.com/bazel-contrib/rules_cuda",
)
load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")
rules_cuda_dependencies()
register_detected_cuda_toolchains()
