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
  base_exe = ctx.file.data.short_path
  exe = ctx.actions.declare_file(ctx.label.name)
  script = SANITIZER_SCRIPT.format(
    sanitizer = sanitizer,
    exe = base_exe
  )
  ctx.actions.write(exe, script, is_executable=True)
  runfiles = ctx.runfiles(files = [ctx.file.data])
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
    "data" : attr.label(
      mandatory = True,
      allow_single_file = True,
    ),
  },
  toolchains = [
    "@rules_cuda//cuda:toolchain_type",
  ],
)
