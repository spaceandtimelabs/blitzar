load("@rules_cuda//cuda/private:toolchain.bzl", "find_cuda_toolkit")

SANITIZER_SCRIPT = """
outfile=$(mktemp)
{sanitizer} --log-file $outfile {exe}
rc=$?
output=$(<$outfile)
if [[ $output =~ "Target application terminated before first instrumented API call" ]]; then
  exit 0
fi
echo $output
rm $outfile
exit $rc
"""

def _compute_sanitize_test_impl(ctx):
  info = find_cuda_toolkit(ctx)
  sanitizer = info.path + "/bin/compute-sanitizer"
  if len(ctx.files.data) != 1:
    fail("must provide a single data file")
  base_exe = list(ctx.files.data)[0].short_path
  exe = ctx.actions.declare_file(ctx.label.name)
  script = SANITIZER_SCRIPT.format(
    sanitizer = sanitizer,
    exe = base_exe
  )
  ctx.actions.write(exe, script, is_executable=True)
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
