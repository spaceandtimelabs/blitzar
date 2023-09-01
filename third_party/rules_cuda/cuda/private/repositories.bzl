"""Generate `@local_cuda//`"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def _to_forward_slash(s):
    return s.replace("\\", "/")

def _is_linux(ctx):
    return ctx.os.name.startswith("linux")

def _is_windows(ctx):
    return ctx.os.name.lower().startswith("windows")

def _get_nvcc_version(repository_ctx, cuda_path):
    result = repository_ctx.execute([cuda_path + "/bin/nvcc", "--version"])
    if result.return_code != 0:
        return [-1, -1]
    for line in [line for line in result.stdout.split("\n") if ", release " in line]:
        segments = line.split(", release ")
        if len(segments) != 2:
            continue
        version = [int(v) for v in segments[-1].split(", ")[0].split(".")]
        if len(version) >= 2:
            return version[:2]
    return [-1, -1]

def detect_cuda_toolkit(repository_ctx):
    """Detect CUDA Toolkit.

    The path to CUDA Toolkit is determined as:
      - the value of `toolkit_path` passed to local_cuda as an attribute
      - taken from `CUDA_PATH` environment variable or
      - determined through 'which ptxas' or
      - defaults to '/usr/local/cuda'

    Args:
        repository_ctx: repository_ctx

    Returns:
        A struct contains the information of CUDA Toolkit.
    """
    cuda_path = repository_ctx.attr.toolkit_path
    if cuda_path == "":
        cuda_path = repository_ctx.os.environ.get("CUDA_PATH", None)
    if cuda_path == None:
        ptxas_path = repository_ctx.which("ptxas")
        if ptxas_path:
            # ${CUDA_PATH}/bin/ptxas

            # Some distributions instead put CUDA binaries in a seperate path
            # Manually check and redirect there when necessary
            alternative = repository_ctx.path("/usr/lib/nvidia-cuda-toolkit/bin/nvcc")
            if str(ptxas_path) == "/usr/bin/ptxas" and alternative.exists:
                ptxas_path = alternative
            cuda_path = str(ptxas_path.dirname.dirname)
    if cuda_path == None and _is_linux(repository_ctx):
        cuda_path = "/usr/local/cuda"

    if cuda_path != None and not repository_ctx.path(cuda_path).exists:
        cuda_path = None

    bin_ext = ".exe" if _is_windows(repository_ctx) else ""
    nvlink = "@rules_cuda//cuda/dummy:nvlink"
    link_stub = "@rules_cuda//cuda/dummy:link.stub"
    bin2c = "@rules_cuda//cuda/dummy:bin2c"
    fatbinary = "@rules_cuda//cuda/dummy:fatbinary"
    if cuda_path != None:
        if repository_ctx.path(cuda_path + "/bin/nvlink" + bin_ext).exists:
            nvlink = str(Label("@local_cuda//:cuda/bin/nvlink{}".format(bin_ext)))
        if repository_ctx.path(cuda_path + "/bin/crt/link.stub").exists:
            link_stub = str(Label("@local_cuda//:cuda/bin/crt/link.stub"))
        if repository_ctx.path(cuda_path + "/bin/bin2c" + bin_ext).exists:
            bin2c = str(Label("@local_cuda//:cuda/bin/bin2c{}".format(bin_ext)))
        if repository_ctx.path(cuda_path + "/bin/fatbinary" + bin_ext).exists:
            fatbinary = str(Label("@local_cuda//:cuda/bin/fatbinary{}".format(bin_ext)))

    nvcc_version_major = -1
    nvcc_version_minor = -1

    if cuda_path != None:
        nvcc_version_major, nvcc_version_minor = _get_nvcc_version(repository_ctx, cuda_path)

    return struct(
        path = cuda_path,
        # this should have been extracted from cuda.h, reuse nvcc for now
        version_major = nvcc_version_major,
        version_minor = nvcc_version_minor,
        # this is extracted from `nvcc --version`
        nvcc_version_major = nvcc_version_major,
        nvcc_version_minor = nvcc_version_minor,
        nvlink_label = nvlink,
        link_stub_label = link_stub,
        bin2c_label = bin2c,
        fatbinary_label = fatbinary,
    )

def config_cuda_toolkit_and_nvcc(repository_ctx, cuda):
    """Generate `@local_cuda//BUILD` and `@local_cuda//defs.bzl` and `@local_cuda//toolchain/BUILD`

    Args:
        repository_ctx: repository_ctx
        cuda: The struct returned from detect_cuda_toolkit
    """

    # Generate @local_cuda//BUILD and @local_cuda//defs.bzl
    defs_bzl_content = ""
    defs_if_local_cuda = "def if_local_cuda(if_true, if_false = []):\n    return %s\n"
    if cuda.path != None:
        # When using a special cuda toolkit path install, need to manually fix up the lib64 links
        if cuda.path == "/usr/lib/nvidia-cuda-toolkit":
            repository_ctx.symlink(cuda.path + "/bin", "cuda/bin")
            repository_ctx.symlink("/usr/lib/x86_64-linux-gnu", "cuda/lib64")
        else:
            repository_ctx.symlink(cuda.path, "cuda")
        repository_ctx.symlink(Label("//cuda:runtime/BUILD.local_cuda"), "BUILD")
        defs_bzl_content += defs_if_local_cuda % "if_true"
    else:
        repository_ctx.file("BUILD")  # Empty file
        defs_bzl_content += defs_if_local_cuda % "if_false"
    repository_ctx.file("defs.bzl", defs_bzl_content)

    # Generate @local_cuda//toolchain/BUILD
    tpl_label = Label(
        "//cuda:templates/BUILD.local_toolchain_" +
        ("nvcc" if _is_linux(repository_ctx) else "nvcc_msvc"),
    )
    substitutions = {
        "%{cuda_path}": _to_forward_slash(cuda.path) if cuda.path else "cuda-not-found",
        "%{cuda_version}": "{}.{}".format(cuda.version_major, cuda.version_minor),
        "%{nvcc_version_major}": str(cuda.nvcc_version_major),
        "%{nvcc_version_minor}": str(cuda.nvcc_version_minor),
        "%{nvlink_label}": cuda.nvlink_label,
        "%{link_stub_label}": cuda.link_stub_label,
        "%{bin2c_label}": cuda.bin2c_label,
        "%{fatbinary_label}": cuda.fatbinary_label,
    }
    env_tmp = repository_ctx.os.environ.get("TMP", repository_ctx.os.environ.get("TEMP", None))
    if env_tmp != None:
        substitutions["%{env_tmp}"] = _to_forward_slash(env_tmp)
    repository_ctx.template("toolchain/BUILD", tpl_label, substitutions = substitutions, executable = False)

def detect_clang(repository_ctx):
    """Detect local clang installation.

    The path to clang is determined by:

      - taken from `CUDA_CLANG_PATH` environment variable or
      - taken from `BAZEL_LLVM` environment variable as `$BAZEL_LLVM/bin/clang` or
      - determined through `which clang` or
      - treated as being not detected and not configured

    Args:
        repository_ctx: repository_ctx

    Returns:
        clang_path (str | None): Optionally return a string of path to clang executable if detected.
    """
    bin_ext = ".exe" if _is_windows(repository_ctx) else ""
    clang_path = repository_ctx.os.environ.get("CUDA_CLANG_PATH", None)
    if clang_path == None:
        bazel_llvm = repository_ctx.os.environ.get("BAZEL_LLVM", None)
        if bazel_llvm != None and repository_ctx.path(bazel_llvm + "/bin/clang" + bin_ext).exists:
            clang_path = bazel_llvm + "/bin/clang" + bin_ext
    if clang_path == None:
        clang_path = str(repository_ctx.which("clang"))

    if clang_path != None and not repository_ctx.path(clang_path).exists:
        clang_path = None

    return clang_path

def config_clang(repository_ctx, cuda, clang_path):
    """Generate `@local_cuda//toolchain/clang/BUILD`

    Args:
        repository_ctx: repository_ctx
        cuda: The struct returned from `detect_cuda_toolkit`
        clang_path: Path to clang executable returned from `detect_clang`
    """
    tpl_label = Label("//cuda:templates/BUILD.local_toolchain_clang")
    substitutions = {
        "%{clang_path}": _to_forward_slash(clang_path) if clang_path else "cuda-clang-not-found",
        "%{cuda_path}": _to_forward_slash(cuda.path) if cuda.path else "cuda-not-found",
        "%{cuda_version}": "{}.{}".format(cuda.version_major, cuda.version_minor),
        "%{nvlink_label}": cuda.nvlink_label,
        "%{link_stub_label}": cuda.link_stub_label,
        "%{bin2c_label}": cuda.bin2c_label,
        "%{fatbinary_label}": cuda.fatbinary_label,
    }
    repository_ctx.template("toolchain/clang/BUILD", tpl_label, substitutions = substitutions, executable = False)

def _local_cuda_impl(repository_ctx):
    cuda = detect_cuda_toolkit(repository_ctx)
    config_cuda_toolkit_and_nvcc(repository_ctx, cuda)

    clang_path = detect_clang(repository_ctx)
    config_clang(repository_ctx, cuda, clang_path)

local_cuda = repository_rule(
    implementation = _local_cuda_impl,
    attrs = {"toolkit_path": attr.string(mandatory = False)},
    configure = True,
    local = True,
    environ = ["CUDA_PATH", "PATH", "CUDA_CLANG_PATH", "BAZEL_LLVM"],
    # remotable = True,
)

def rules_cuda_dependencies(toolkit_path = None):
    """Populate the dependencies for rules_cuda. This will setup workspace dependencies (other bazel rules) and local toolchains.

    Args:
        toolkit_path: Optionally specify the path to CUDA toolkit. If not specified, it will be detected automatically.
    """
    maybe(
        name = "bazel_skylib",
        repo_rule = http_archive,
        sha256 = "f7be3474d42aae265405a592bb7da8e171919d74c16f082a5457840f06054728",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",
        ],
    )

    maybe(
        name = "platforms",
        repo_rule = http_archive,
        sha256 = "379113459b0feaf6bfbb584a91874c065078aa673222846ac765f86661c27407",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.5/platforms-0.0.5.tar.gz",
            "https://github.com/bazelbuild/platforms/releases/download/0.0.5/platforms-0.0.5.tar.gz",
        ],
    )

    local_cuda(name = "local_cuda", toolkit_path = toolkit_path)
