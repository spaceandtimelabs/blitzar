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

# Rules CUDA version that I forked.
# http_archive(
#     name = "rules_cuda",
#     sha256 = "dc1f4f704ca56e3d5edd973f98a45f0487d0f28c689d0a57ba236112148b1833",
#     strip_prefix = "rules_cuda-v0.1.2",
#     urls = ["https://github.com/bazel-contrib/rules_cuda/releases/download/v0.1.2/rules_cuda-v0.1.2.tar.gz"],
# )

# Rules CUDA fork with .cc
# I couldn't get bazel to input this repo without a build error.
# http_archive(
#     name = "local_config_cuda",
#     sha256 = "90ee8222c86bf8addc685a603c6fb294b5bd85b79ea981118a75344d6246504e",
#     urls = [
#         "https://github.com/jacobtrombetta/rules_cuda/archive/refs/heads/main.tar.gz",
#     ],
# )

local_repository(
    name = "rules_cuda",
    path = "third_party/rules_cuda",
)

load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")
rules_cuda_dependencies()
register_detected_cuda_toolchains()

# load("@build_bazel_rules_cuda//gpus:cuda_configure.bzl", "cuda_configure")

# cuda_configure(name = "local_config_cuda")

git_repository(
    name = "com_github_catchorg_catch2",
    commit = "6f21a3609cea360846a0ca93be55877cca14c86d",
    remote = "https://github.com/catchorg/Catch2",
)
