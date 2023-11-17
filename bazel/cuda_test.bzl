load("@rules_cuda//cuda/private:toolchain.bzl", "find_cuda_toolkit")

def _compute_sanitize_test_impl(ctx):
  info = find_cuda_toolkit(ctx)
  sanitize = info.path + "/bin/compute-sanitizer"
  if len(ctx.files.data) != 1:
    fail("must provide a single data file")
  base_exe = list(ctx.files.data)[0].path
  exe = ctx.actions.declare_file(ctx.label.name)
  ctx.actions.write(exe, "%s %s" % (sanitize, base_exe), is_executable=True)
  runfiles = ctx.runfiles(files = ctx.files.data)
  transitive_runfiles = []
  for runfiles_attr in ( ctx.attr.data, ):
    for target in runfiles_attr:
      transitive_runfiles.append(target[DefaultInfo].default_runfiles)
  runfiles = runfiles.merge_all(transitive_runfiles)
  return [
    DefaultInfo(
      executable = exe,
      runfiles = runfiles,
    ),
  ]

compute_sanitize_test = rule(
  implementation = _compute_sanitize_test_impl,
  test = True,
  attrs = {
    "data" : attr.label_list(
      allow_files = True,
    ),
  },
  toolchains = [
    "@rules_cuda//cuda:toolchain_type",
  ],
)
