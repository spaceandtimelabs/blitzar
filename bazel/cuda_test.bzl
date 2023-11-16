load("@rules_cuda//cuda/private:toolchain.bzl", "find_cuda_toolkit")

def _cuda_test_impl(ctx):
  # info = ctx.toolchains['@rules_cuda//cuda:toolchain_type'].cuda_toolkit
  info = find_cuda_toolkit(ctx)
  # print(info)
  sanitize = info.path + "/bin/compute-sanitizer"
  # fail("arf: " + sanitize)
  exe = ctx.actions.declare_file(ctx.label.name)
  ctx.actions.write(exe, "{} --help".format(sanitize), is_executable=True)
  return [
    DefaultInfo(executable = exe)
  ]

cuda_test = rule(
  implementation = _cuda_test_impl,
  test = True,
  attrs = {
  },
  toolchains = [
    "@rules_cuda//cuda:toolchain_type",
  ],
)
