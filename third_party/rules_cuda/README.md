# CUDA rules for [Bazel](https://bazel.build)

This repository contains [Starlark](https://github.com/bazelbuild/starlark) implementation of CUDA rules in Bazel.

These rules provide some macros and rules that make it easier to build CUDA with Bazel.

## Getting Started

Add the following to your `WORKSPACE` file and replace the placeholders with actual values.

```starlark
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "rules_cuda",
    sha256 = "{sha256_to_replace}",
    strip_prefix = "rules_cuda-{git_commit_hash}",
    urls = ["https://github.com/bazel-contrib/rules_cuda/archive/{git_commit_hash}.tar.gz"],
)
load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")
rules_cuda_dependencies()
register_detected_cuda_toolchains()
```

**NOTE**: the use of `register_detected_cuda_toolchains` depends on the environment variable `CUDA_PATH`. You must also
ensure the host compiler is available. On windows, this means that you will also need to set the environment variable
`BAZEL_VC` properly.

[`detect_cuda_toolkit`](https://github.com/bazel-contrib/rules_cuda/blob/5633f0c0f7/cuda/private/repositories.bzl#L28-L58)
and [`detect_clang`](https://github.com/bazel-contrib/rules_cuda/blob/5633f0c0f7/cuda/private/repositories.bzl#L143-L166)
determains how the toolchains are detected.

### Rules

- `cuda_library`: Can be used to compile and create static library for CUDA kernel code. The resulting targets can be
  consumed by [C/C++ Rules](https://bazel.build/reference/be/c-cpp#rules).
- `cuda_objects`: If you don't understand what _device link_ means, you must never use it. This rule produce incomplete
  object files that can only be consumed by `cuda_library`. It is created for relocatable device code and device link
  time optimization source files.

### Flags

Some flags are defined in [cuda/BUILD.bazel](cuda/BUILD.bazel). To use them, for example:

```
bazel build --@rules_cuda//cuda:archs=compute_61:compute_61,sm_61
```

In `.bazelrc` file, you can define shortcut alias for the flag, for example:

```
# Convenient flag shortcuts.
build --flag_alias=cuda_archs=@rules_cuda//cuda:archs
```

and then you can use it as following:

```
bazel build --cuda_archs=compute_61:compute_61,sm_61
```

#### Available flags

- `@rules_cuda//cuda:enable`

  Enable or disable all rules_cuda related rules. When disabled, the detected cuda toolchains will also be disabled to avoid potential human error.
  By default, rules_cuda rules are enabled. See `examples/if_cuda` for how to support both cuda-enabled and cuda-free builds.

- `@rules_cuda//cuda:archs`

  Select the cuda archs to support. See [cuda_archs specification DSL grammar](https://github.com/bazel-contrib/rules_cuda/blob/5633f0c0f7/cuda/private/rules/flags.bzl#L14-L44).

- `@rules_cuda//cuda:compiler`

  Select the cuda compiler, available options are `nvcc` or `clang`

- `@rules_cuda//cuda:copts`

  Add the copts to all cuda compile actions.

- `@rules_cuda//cuda:host_copts`

  Add the copts to the host compiler.

- `@rules_cuda//cuda:runtime`

  Set the default cudart to link, for example, `--@rules_cuda//cuda:runtime=@local_cuda//:cuda_runtime_static` link the static cuda runtime.

- `--features=cuda_device_debug`

  Sets nvcc flags to enable debug information in device code.
  Currently ignored for clang, where `--compilation_mode=debug` applies to both
  host and device code.

## Examples

Checkout the examples to see if it fits your needs.

See [examples](./examples) for basic usage.

See [rules_cuda_examples](https://github.com/cloudhan/rules_cuda_examples) for extended real world projects.

## Known issue

Sometimes the following error occurs:

```
cc1plus: fatal error: /tmp/tmpxft_00000002_00000019-2.cpp: No such file or directory
```

The problem is caused by nvcc use PID to determine temporary file name, and with `--spawn_strategy linux-sandbox` which is the default strategy on Linux, the PIDs nvcc sees are all very small numbers, say 2~4 due to sandboxing. `linux-sandbox` is not hermetic because [it mounts root into the sandbox](https://docs.bazel.build/versions/main/command-line-reference.html#flag--experimental_use_hermetic_linux_sandbox), thus, `/tmp` is shared between sandboxes, which is causing name conflict under high parallelism. Similar problem has been reported at [nvidia forums](https://forums.developer.nvidia.com/t/avoid-generating-temp-files-in-tmp-while-nvcc-compiling/197657/10).

To avoid it:

- Use `--spawn_strategy local` should eliminate the case because it will let nvcc sees the true PIDs.
- Use `--experimental_use_hermetic_linux_sandbox` should eliminate the case because it will avoid the sharing of `/tmp`.
- Add `-objtemp` option to the command should reduce the case from happening.
