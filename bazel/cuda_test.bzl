def _cuda_test_impl(ctx):
  exe = ctx.actions.declare_file(ctx.label.name)
  ctx.actions.write(exe, "", is_executable=True)
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
