{ pkgs, clang, name }:
# TODO(rnburn): Modify the build so that the driver will add these arguments automatically
# and then we can drop this wrapper.
pkgs.writers.writePython3Bin "${name}"
{
  flakeIgnore = [ "E501" ];
} ''
  import sys
  import os
  args_p = ['-stdlib=libc++'] + sys.argv[1:]
  args_p = ['-isystem', '${clang}/include/x86_64-unknown-linux-gnu/c++/v1'] + args_p
  args_p = ['-L', '${clang}/lib/x86_64-unknown-linux-gnu'] + args_p
  compiler = '${clang}/bin/${name}'
  os.execv(compiler, [compiler] + args_p)
''
