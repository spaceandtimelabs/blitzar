#!/usr/bin/env python3
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Crosstool wrapper for compiling CUDA programs.

SYNOPSIS:
  crosstool_wrapper_is_not_gcc [options passed in by cc_library()
                                or cc_binary() rule]

DESCRIPTION:
  This script is expected to be called by the cc_library() or cc_binary() bazel
  rules. When the option "-x cuda" is present in the list of arguments passed
  to this script, it invokes the nvcc CUDA compiler. Most arguments are passed
  as is as a string to --compiler-options of nvcc. When "-x cuda" is not
  present, this wrapper invokes hybrid_driver_is_not_gcc with the input
  arguments as is.

NOTES:
  Changes to the contents of this file must be propagated from
  //third_party/gpus/crosstool/crosstool_wrapper_is_not_gcc to
  //third_party/gpus/crosstool/v*/*/clang/bin/crosstool_wrapper_is_not_gcc
"""

from __future__ import print_function

__author__ = 'keveman@google.com (Manjunath Kudlur)'

from argparse import ArgumentParser
import contextlib
import os
import subprocess
import re
import sys
import pipes
import tempfile

# Template values set by cuda_autoconf.
CPU_COMPILER = ('%{cpu_compiler}')
CPU_CXX_COMPILER = ('%{cpu_cxx_compiler}')
GCC_HOST_COMPILER_PATH = ('%{gcc_host_compiler_path}')

NVCC_PATH = '%{nvcc_path}'
PREFIX_DIR = os.path.dirname(GCC_HOST_COMPILER_PATH)
NVCC_VERSION = '%{cuda_version}'

def Log(s):
  print('gpus/crosstool: {0}'.format(s))

@contextlib.contextmanager
def ClosingFileDescriptor(fd):
  try:
    yield fd
  finally:
    os.close(fd)


def NormalizeArgs(args):
  result = []
  for arg in args:
    if arg.startswith('@'):
      with open(arg[1:], 'r') as f:
        result += [line[:-1] for line in f.readlines()]
    else:
      result.append(arg)
  return result


def GetOptionValue(argv, option):
  """Extract the list of values for option from the argv list.

  Args:
    argv: A list of strings, possibly the argv passed to main().
    option: The option whose value to extract, with the leading '-'.

  Returns:
    A list of values, either directly following the option,
    (eg., -opt val1 val2) or values collected from multiple occurrences of
    the option (eg., -opt val1 -opt val2).
  """

  parser = ArgumentParser()
  parser.add_argument(option, nargs='*', action='append')
  option = option.lstrip('-').replace('-', '_')
  args, _ = parser.parse_known_args(argv)
  if not args or not vars(args)[option]:
    return []
  else:
    return sum(vars(args)[option], [])


def GetHostCompilerOptions(argv):
  """Collect the -isystem, -iquote, and --sysroot option values from argv.

  Args:
    argv: A list of strings, possibly the argv passed to main().

  Returns:
    The string that can be used as the --compiler-options to nvcc.
  """

  parser = ArgumentParser()
  parser.add_argument('-isystem', nargs='*', action='append')
  parser.add_argument('-iquote', nargs='*', action='append')
  parser.add_argument('--sysroot', nargs=1)
  parser.add_argument('-g', nargs='*', action='append')
  parser.add_argument('-fno-canonical-system-headers', action='store_true')
  parser.add_argument('-no-canonical-prefixes', action='store_true')

  args, _ = parser.parse_known_args(argv)

  opts = ''

  if args.isystem:
    opts += ' -isystem ' + ' -isystem '.join(sum(args.isystem, []))
  if args.iquote:
    opts += ' -iquote ' + ' -iquote '.join(sum(args.iquote, []))
  if args.g:
    opts += ' -g' + ' -g'.join(sum(args.g, []))
  if args.fno_canonical_system_headers:
    opts += ' -fno-canonical-system-headers'
  if args.no_canonical_prefixes:
    opts += ' -no-canonical-prefixes'
  if args.sysroot:
    opts += ' --sysroot ' + args.sysroot[0]

  return opts

def _update_options(nvcc_options):
  if NVCC_VERSION in ("7.0",):
    return nvcc_options

  update_options = { "relaxed-constexpr" : "expt-relaxed-constexpr" }
  return [ update_options[opt] if opt in update_options else opt
                    for opt in nvcc_options ]

def GetNvccOptions(argv):
  """Collect the -nvcc_options values from argv.

  Args:
    argv: A list of strings, possibly the argv passed to main().

  Returns:
    The string that can be passed directly to nvcc.
  """

  parser = ArgumentParser()
  parser.add_argument('-nvcc_options', nargs='*', action='append')

  args, _ = parser.parse_known_args(argv)

  if args.nvcc_options:
    options = _update_options(sum(args.nvcc_options, []))
    return ' '.join(['--'+a for a in options])
  return ''

def system(cmd):
  """Invokes cmd with os.system().

  Args:
    cmd: The command.

  Returns:
    The exit code if the process exited with exit() or -signal
    if the process was terminated by a signal.
  """
  retv = os.system(cmd)
  if os.WIFEXITED(retv):
    return os.WEXITSTATUS(retv)
  else:
    return -os.WTERMSIG(retv)

def CompileNvcc(argv, log=False, device_c=False):
  """Compile with nvcc using arguments assembled from argv.

  Args:
    argv: A list of strings, possibly the argv passed to main().
    log: True if logging is requested.

  Returns:
    The return value of calling system('nvcc ' + args)
  """

  host_compiler_options = GetHostCompilerOptions(argv)
  host_compiler_options += ' -fcoroutines'
  nvcc_compiler_options = GetNvccOptions(argv)
  opt_option = GetOptionValue(argv, '-O')
  m_options = GetOptionValue(argv, '-m')
  m_options = ''.join([' -m' + m for m in m_options if m in ['32', '64']])
  include_options = GetOptionValue(argv, '-I')
  out_file = GetOptionValue(argv, '-o')
  depfiles = GetOptionValue(argv, '-MF')
  defines = GetOptionValue(argv, '-D')
  defines = ''.join([' -D' + define for define in defines])
  undefines = GetOptionValue(argv, '-U')
  undefines = ''.join([' -U' + define for define in undefines])
  std_options = GetOptionValue(argv, '-std')
  # Supported -std flags as of CUDA 9.0. Only keep last to mimic gcc/clang.
  nvcc_allowed_std_options = ["c++03", "c++11", "c++14", "c++17", "c++20"]
  std_options = ''.join([' -std=' + define
      for define in std_options if define in nvcc_allowed_std_options][-1:])
  fatbin_options = ''.join([' --fatbin-options=' + option
      for option in GetOptionValue(argv, '-Xcuda-fatbinary')])

  # The list of source files get passed after the -c option. I don't know of
  # any other reliable way to just get the list of source files to be compiled.
  src_files = GetOptionValue(argv, '-c')

  # Pass -w through from host to nvcc, but don't do anything fancier with
  # warnings-related flags, since they're not necessarily the same across
  # compilers.
  warning_options = ' -w' if '-w' in argv else ''

  if len(src_files) == 0:
    return 1
  if len(out_file) != 1:
    return 1

  opt = (' -O2' if (len(opt_option) > 0 and int(opt_option[0]) > 0)
         else ' -g')

  includes = (' -I ' + ' -I '.join(include_options)
              if len(include_options) > 0
              else '')

  # Unfortunately, there are other options that have -c prefix too.
  # So allowing only those look like C/C++ files.
  src_files = [f for f in src_files if
               re.search('\.cpp$|\.cc$|\.cu$|\.c$|\.cxx$|\.C$', f)]
  srcs = ' '.join(src_files)
  out = ' -o ' + out_file[0]

  # We add this -std=c++17 flag, because
  # benchmarks could not be compiled without it.
  # The `build --cxxopt -std=c++17` flag set in the 
  # `.bazelrc` file was not passed to the compiler.
  # However, this flag is relevant to some modules.
  nvccopts = '-std=c++20 '

  nvccopts += '-D_FORCE_INLINES '
  for capability in GetOptionValue(argv, "--cuda-gpu-arch"):
    capability = capability[len('sm_'):]
    nvccopts += r'-gencode=arch=compute_%s,\"code=sm_%s\" ' % (capability,
                                                               capability)
  for capability in GetOptionValue(argv, '--cuda-include-ptx'):
    capability = capability[len('sm_'):]
    nvccopts += r'-gencode=arch=compute_%s,\"code=compute_%s\" ' % (capability,
                                                                    capability)
  nvccopts += nvcc_compiler_options
  nvccopts += ' --keep' # Doesn't actively remove files under /tmp, it causes compilation errors sometimes.
  nvccopts += ' --allow-unsupported-compiler' # Allow any version of clang / gcc.
  nvccopts += undefines
  nvccopts += defines
  nvccopts += std_options
  nvccopts += m_options
  nvccopts += warning_options
  nvccopts += fatbin_options
  nvccopts += ' --extended-lambda'

  if device_c:
    nvccopts += ' --device-c'

  if depfiles:
    # Generate the dependency file
    depfile = depfiles[0]
    cmd = (NVCC_PATH + ' ' + nvccopts +
           ' --compiler-options "' + host_compiler_options + '"' +
           ' --compiler-bindir=' + GCC_HOST_COMPILER_PATH +
           ' -I .' +
           ' -x cu ' + opt + includes + ' ' + srcs + ' -M -o ' + depfile)
    if log: Log(cmd)
    exit_status = system(cmd)
    if exit_status != 0:
      return exit_status

  cmd = (NVCC_PATH + ' ' + nvccopts +
         ' --compiler-options "' + host_compiler_options + ' -fPIC"' +
         ' --compiler-bindir=' + GCC_HOST_COMPILER_PATH +
         ' -I .' +
         ' -x cu ' + opt + includes + ' -c ' + srcs + out)

  # TODO(zhengxq): for some reason, 'gcc' needs this help to find 'as'.
  # Need to investigate and fix.
  cmd = 'PATH=' + PREFIX_DIR + ':$PATH ' + cmd
  if log: Log(cmd)
  return system(cmd)

def ProcessLinkArgs(args_fd, argv):
  nargs = len(argv)
  args = []
  index = 0
  while index < nargs:
    arg = argv[index]
    if arg == '-o' and index + 1 < nargs:
      args += ['-o', argv[index+1]]
      index += 1
    elif arg.startswith('-x'):
      index += 1
      pass
    elif arg == '--cudalog':
      pass
    elif arg == '-dlink':
      args.append(arg)
    elif arg.startswith('-'):
      os.write(args_fd, str.encode(arg + '\n'))
    else:
      args.append(arg)
    index += 1
  return args

def LinkNvcc(argv, log=False):
  """Link with nvcc using arguments assembled from argv.
  
  Args:
    argv: A list of strings, possibly the argv passed to main().
    log: True if logging is requested.
  """
  args_fd, args_path = tempfile.mkstemp(dir='./', suffix='.params')
  with ClosingFileDescriptor(args_fd):
    args = ProcessLinkArgs(args_fd, argv)
  args = [
    '--compiler-options',
    '@%s' % args_path
  ] + args
  # Work around for silencing nvlink warnings
  args.append('--gpu-architecture=sm_70')
  cmd = [NVCC_PATH] + args
  return subprocess.call(cmd)

def SanitizeFlagfile(in_path, out_fd):
  with open(in_path, "r") as in_fp:
    for line in in_fp:
      if line != "-lstdc++\n":
        os.write(out_fd, bytearray(line, 'utf8'))

def RewriteStaticLinkArgs(argv):
  prev_argv = list(argv)
  argv = []
  for arg in prev_argv:
    if arg == "-lstdc++":
      pass
    elif arg.startswith("-Wl,@"):
      # tempfile.mkstemp will write to the out-of-sandbox tempdir
      # unless the user has explicitly set environment variables
      # before starting Bazel. But here in $PWD is the Bazel sandbox,
      # which will be deleted automatically after the compiler exits.
      (flagfile_fd, flagfile_path) = tempfile.mkstemp(
          dir='./', suffix=".linker-params")
      with ClosingFileDescriptor(flagfile_fd):
        SanitizeFlagfile(arg[len("-Wl,@"):], flagfile_fd)
      argv.append("-Wl,@" + flagfile_path)
    elif arg.startswith("@"):
      # tempfile.mkstemp will write to the out-of-sandbox tempdir
      # unless the user has explicitly set environment variables
      # before starting Bazel. But here in $PWD is the Bazel sandbox,
      # which will be deleted automatically after the compiler exits.
      (flagfile_fd, flagfile_path) = tempfile.mkstemp(
          dir='./', suffix=".linker-params")
      with ClosingFileDescriptor(flagfile_fd):
        SanitizeFlagfile(arg[len("@"):], flagfile_fd)
      argv.append("@" + flagfile_path)
    else:
      argv.append(arg)
  return argv

def main():
  parser = ArgumentParser()
  parser.add_argument('-x', nargs=1)
  parser.add_argument('--cuda_log', action='store_true')
  parser.add_argument('--device-c', dest='device_c', action='store_true')
  normalized_args = NormalizeArgs(sys.argv[1:])

  args, leftover = parser.parse_known_args(normalized_args)

  if args.x and args.x[0] == 'cuda':
    if args.cuda_log: Log('-x cuda')
    leftover = [pipes.quote(s) for s in leftover]
    if args.cuda_log: Log('using nvcc')
    if '-c' in leftover:
      return CompileNvcc(leftover, log=args.cuda_log, device_c=args.device_c)
    else:
      return LinkNvcc(normalized_args, log=args.cuda_log)

  # Strip our flags before passing through to the CPU compiler for files which
  # are not -x cuda. We can't just pass 'leftover' because it also strips -x.
  # We not only want to pass -x to the CPU compiler, but also keep it in its
  # relative location in the argv list (the compiler is actually sensitive to
  # this).
  cpu_compiler_flags = [flag for flag in sys.argv[1:]
                             if not flag.startswith(('--cuda_log'))]

  compiler = CPU_COMPILER
  if '-static-libstdc++' in normalized_args:
    compiler = CPU_CXX_COMPILER
    cpu_compiler_flags = RewriteStaticLinkArgs(cpu_compiler_flags)

  return subprocess.call([compiler] + cpu_compiler_flags)

if __name__ == '__main__':
  sys.exit(main())
