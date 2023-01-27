#!/usr/bin/python3

########################################
# Adopted from Envoy
#
# See third_party/license/envoy.LICENSE
########################################

import argparse
import common
import multiprocessing
import os
import os.path
import pathlib
import re
import subprocess
import stat
import sys
import traceback
import shutil
import paths

BUILDIFIER_PATH = paths.get_buildifier()
CLANG_FORMAT_PATH = os.getenv("CLANG_FORMAT", "clang-format-16")
BUILD_FIXER_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "build_fixer.py")
HEADER_ORDER_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "header_order.py")

SUBDIR_SET = set(common.include_dir_order())

SUFFIXES = ("BUILD", "WORKSPACE", ".bzl", ".cc", ".h")

EXCLUDED_PREFIXES = ("./third_party/", "./.git/", "./bazel-", "./.cache", "./ci", "./rust")

class FormatChecker:
    def __init__(self, args):
        self.target_path = args.target_path
        self.operation_type = args.operation_type
        self.include_dir_order = args.include_dir_order

    # Map a line transformation function across each line of a file,
    # writing the result lines as requested.
    # If there is a clang format nesting or mismatch error, return the first occurrence
    def evaluate_lines(self, path, line_xform, write=True):
        error_message = None
        format_flag = True
        output_lines = []

        for line_number, line in enumerate(self.read_lines(path)):
            if line.find("// clang-format off") != -1:
                if not format_flag and error_message is None:
                    error_message = "%s:%d: %s" % (path, line_number + 1, "clang-format nested off")
                format_flag = False
            if line.find("// clang-format on") != -1:
                if format_flag and error_message is None:
                    error_message = "%s:%d: %s" % (path, line_number + 1, "clang-format nested on")
                format_flag = True
            if format_flag:
                output_lines.append(line_xform(line, line_number))
            else:
                output_lines.append(line)
        # We used to use fileinput in the older Python 2.7 script, but this doesn't do
        # inplace mode and UTF-8 in Python 3, so doing it the manual way.
        if write:
            pathlib.Path(path).write_text('\n'.join(output_lines), encoding='utf-8')
        if not format_flag and error_message is None:
            error_message = "%s:%d: %s" % (path, line_number + 1, "clang-format remains off")
        return error_message

    # Obtain all the lines in a given file.
    def read_lines(self, path):
        return self.read_file(path).split('\n')

    # Read a UTF-8 encoded file as a str.
    def read_file(self, path):
        return pathlib.Path(path).read_text(encoding='utf-8')

    # look_path searches for the given executable in all directories in PATH
    # environment variable. If it cannot be found, empty string is returned.
    def look_path(self, executable):
        if executable is None:
            return ''
        return shutil.which(executable) or ''

    # path_exists checks whether the given path exists. This function assumes that
    # the path is absolute and evaluates environment variables.
    def path_exists(self, executable):
        if executable is None:
            return False
        return os.path.exists(os.path.expandvars(executable))

    # executable_by_others checks whether the given path has execute permission for
    # others.
    def executable_by_others(self, executable):
        st = os.stat(os.path.expandvars(executable))
        return bool(st.st_mode & stat.S_IXOTH)

    # Check whether all needed external tools (clang-format, buildifier) are
    # available.
    def check_tools(self):
        error_messages = []

        clang_format_abs_path = self.look_path(CLANG_FORMAT_PATH)
        if clang_format_abs_path:
            if not self.executable_by_others(clang_format_abs_path):
                error_messages.append(
                    "command {} exists, but cannot be executed by other "
                    "users".format(CLANG_FORMAT_PATH))
        else:
            error_messages.append(
                "Command {} not found. If you have clang-format in version 12.x.x "
                "installed, but the binary name is different or it's not available in "
                "PATH, please use CLANG_FORMAT environment variable to specify the path. "
                "Examples:\n"
                "    export CLANG_FORMAT=clang-format-12.0.0\n"
                "    export CLANG_FORMAT=/opt/bin/clang-format-12\n"
                "    export CLANG_FORMAT=/usr/local/opt/llvm@12/bin/clang-format".format(
                    CLANG_FORMAT_PATH))

        def check_bazel_tool(name, path, var):
            bazel_tool_abs_path = self.look_path(path)
            if bazel_tool_abs_path:
                if not self.executable_by_others(bazel_tool_abs_path):
                    error_messages.append(
                        "command {} exists, but cannot be executed by other "
                        "users".format(path))
            elif self.path_exists(path):
                if not self.executable_by_others(path):
                    error_messages.append(
                        "command {} exists, but cannot be executed by other "
                        "users".format(path))
            else:
                error_messages.append(
                    "Command {} not found. If you have {} installed, but the binary "
                    "name is different or it's not available in $GOPATH/bin, please use "
                    "{} environment variable to specify the path. Example:\n"
                    "    export {}=`which {}`\n"
                    "If you don't have {} installed, you can install it by:\n"
                    "    go get -u github.com/bazelbuild/buildtools/{}".format(
                        path, name, var, var, name, name, name))

        check_bazel_tool('buildifier', BUILDIFIER_PATH, 'BUILDIFIER_BIN')

        return error_messages

    def is_build_file(self, file_path):
        basename = os.path.basename(file_path)

        if basename in {"BUILD", "BUILD.bazel"} or basename.endswith(".BUILD"):
            return True

        return False

    def is_starlark_file(self, file_path):
        return file_path.endswith(".bzl")

    def is_workspace_file(self, file_path):
        return os.path.basename(file_path) == "WORKSPACE"

    def check_file_contents(self, file_path, checker):
        error_messages = []

        def check_format_errors(line, line_number):
            def report_error(message):
                error_messages.append("%s:%d: %s" % (file_path, line_number + 1, message))

            checker(line, file_path, report_error)

        evaluate_failure = self.evaluate_lines(file_path, check_format_errors, False)

        if evaluate_failure is not None:
            error_messages.append(evaluate_failure)

        return error_messages

    def check_source_line(self, line, file_path, report_error):
        # Check fixable errors. These may have been fixed already.
        if line.find(".  ") != -1:
            report_error("over-enthusiastic spaces")

        if " ?: " in line:
            # The ?: operator is non-standard, it is a GCC extension
            report_error("Don't use the '?:' operator, it is a non-standard GCC extension")

        normalized_target_path = file_path

        if not normalized_target_path.startswith("./"):
            normalized_target_path = f"./{normalized_target_path}"

    def fix_build_path(self, file_path):
        error_messages = []

        if not self.is_starlark_file(file_path) and not self.is_workspace_file(file_path):
            if os.system("%s %s %s" % (BUILD_FIXER_PATH, file_path, file_path)) != 0:
                error_messages += ["build_fixer rewrite failed for file: %s" % file_path]

        if os.system("%s -lint=fix -mode=fix %s" % (BUILDIFIER_PATH, file_path)) != 0:
            error_messages += ["buildifier rewrite failed for file: %s" % file_path]

        return error_messages

    def check_build_path(self, file_path):
        error_messages = []

        if not self.is_starlark_file(file_path) and not self.is_workspace_file(file_path):

            command = "%s %s | diff %s -" % (BUILD_FIXER_PATH, file_path, file_path)

            error_messages += self.execute_command(
                command, "build_fixer check failed", file_path)

        command = "%s -mode=diff %s" % (BUILDIFIER_PATH, file_path)
        error_messages += self.execute_command(command, "buildifier check failed", file_path)

        return error_messages

    def fix_source_path(self, file_path):
        error_messages = []

        error_messages += self.fix_header_order(file_path)
        error_messages += self.clang_format(file_path)

        return error_messages

    def check_source_path(self, file_path):
        error_messages = self.check_file_contents(file_path, self.check_source_line)

        command = (
            "%s --include_dir_order %s --path %s | diff %s -" %
            (HEADER_ORDER_PATH, self.include_dir_order, file_path, file_path))

        error_messages += self.execute_command(
            command, "header_order.py check failed", file_path)

        command = ("%s %s | diff %s -" % (CLANG_FORMAT_PATH, file_path, file_path))
        error_messages += self.execute_command(command, "clang-format check failed", file_path)

        return error_messages

    def execute_command(
        self,
        command,
        error_message,
        file_path,
        regex=re.compile(r"^(\d+)[a|c|d]?\d*(?:,\d+[a|c|d]?\d*)?$")):

        try:
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT).strip()
            if output:
                return output.decode('utf-8').split("\n")
            return []
        except subprocess.CalledProcessError as e:
            if (e.returncode != 0 and e.returncode != 1):
                return ["ERROR: something went wrong while executing: %s" % e.cmd]
            # In case we can't find any line numbers, record an error message first.
            error_messages = ["%s for file: %s" % (error_message, file_path)]
            for line in e.output.decode('utf-8').splitlines():
                for num in regex.findall(line):
                    error_messages.append("  %s:%s" % (file_path, num))
            return error_messages

    def fix_header_order(self, file_path):
        command = "%s --rewrite --include_dir_order %s --path %s" % (
            HEADER_ORDER_PATH, self.include_dir_order, file_path)

        if os.system(command) != 0:
            return ["header_order.py rewrite error: %s" % (file_path)]

        return []

    def clang_format(self, file_path):
        command = "%s -i %s" % (CLANG_FORMAT_PATH, file_path)

        if os.system(command) != 0:
            return ["clang-format rewrite error: %s" % (file_path)]

        return []

    def check_format(self, file_path):
        error_messages = []
        # Apply fixes first, if asked, and then run checks. If we wind up attempting to fix
        # an issue, but there's still an error, that's a problem.
        try_to_fix = self.operation_type == "fix"

        if self.is_build_file(file_path) or self.is_starlark_file(
                file_path) or self.is_workspace_file(file_path):
            if try_to_fix:
                error_messages += self.fix_build_path(file_path)

            error_messages += self.check_build_path(file_path)
        else:
            if try_to_fix:
                error_messages += self.fix_source_path(file_path)
            error_messages += self.check_source_path(file_path)

        if error_messages:
            return ["From %s" % file_path] + error_messages
        return error_messages

    def check_format_return_trace_on_error(self, file_path):
        """Run check_format and return the traceback of any exception."""
        try:
            return self.check_format(file_path)
        except:
            return traceback.format_exc().split("\n")

    def check_format_visitor(self, arg, dir_name, names):
        """Run check_format in parallel for the given files.
        Args:
        arg: a tuple (pool, result_list)
            pool and result_list are for starting tasks asynchronously.
        dir_name: the parent directory of the given files.
        names: a list of file names.
        """

        # Unpack the multiprocessing.Pool process pool and list of results. Since
        # python lists are passed as references, this is used to collect the list of
        # async results (futures) from running check_format and passing them back to
        # the caller.
        pool, result_list = arg
        
        dir_name = normalize_path(dir_name)

        for file_name in names:
            result = pool.apply_async(
                self.check_format_return_trace_on_error, args=(dir_name + file_name,))
            result_list.append(result)

    # check_error_messages iterates over the list with error messages and prints
    # errors and returns a bool based on whether there were any errors.
    def check_error_messages(self, error_messages):
        if error_messages:
            for e in error_messages:
                print("ERROR: %s" % e)
            return True
        return False


def normalize_path(path):
    """Convert path to form ./path/to/dir/ for directories and ./path/to/file otherwise"""
    if not path.startswith("./"):
        path = "./" + path

    isdir = os.path.isdir(path)
    if isdir and not path.endswith("/"):
        path += "/"

    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check or fix file format.")

    parser.add_argument(
        "operation_type",
        type=str,
        choices=["check", "fix"],
        help="specify if the run should 'check' or 'fix' format.")

    parser.add_argument(
        "target_path",
        type=str,
        nargs="?",
        default=".",
        help="specify the root directory for the script to recurse over. Default '.'.")

    parser.add_argument(
        "-j",
        "--num-workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="number of worker processes to use; defaults to one per core.")

    parser.add_argument(
        "--include_dir_order",
        type=str,
        default=",".join(common.include_dir_order()),
        help="specify the header block include directory order.")

    args = parser.parse_args()

    format_checker = FormatChecker(args)

    # Check whether all needed external tools are available.
    ct_error_messages = format_checker.check_tools()

    if format_checker.check_error_messages(ct_error_messages):
        sys.exit(1)

    error_messages = []
    
    if os.path.isfile(args.target_path):
        # All of our EXCLUDED_PREFIXES start with "./", but the provided
        # target path argument might not. Add it here if it is missing,
        # and use that normalized path for both lookup and `check_format`.
        normalized_target_path = normalize_path(args.target_path)

        if not normalized_target_path.startswith(
                EXCLUDED_PREFIXES) and normalized_target_path.endswith(SUFFIXES):
            error_messages += format_checker.check_format(normalized_target_path)
    else:
        results = []

        def pooled_check_format(path_predicate):
            pool = multiprocessing.Pool(processes=args.num_workers)
            # For each file in target_path, start a new task in the pool and collect the
            # results (results is passed by reference, and is used as an output).
            for root, _, files in os.walk(args.target_path):
                _files = []

                for filename in files:
                    file_path = os.path.join(root, filename)
                    check_file = (
                        path_predicate(filename) and not file_path.startswith(EXCLUDED_PREFIXES)
                        and file_path.endswith(SUFFIXES))
                    if check_file:
                        _files.append(filename)
                if not _files:
                    continue

                format_checker.check_format_visitor((pool, results), root, _files)

            # Close the pool to new tasks, wait for all of the running tasks to finish,
            # then collect the error messages.
            pool.close()
            pool.join()

        # We first run formatting on non-BUILD files, since the BUILD file format
        # requires analysis of srcs/hdrs in the BUILD file, and we don't want these
        # to be rewritten by other multiprocessing pooled processes.
        pooled_check_format(lambda f: not format_checker.is_build_file(f))
        pooled_check_format(lambda f: format_checker.is_build_file(f))

        error_messages += sum((r.get() for r in results), [])

    if format_checker.check_error_messages(error_messages):
        print("ERROR: check format failed. run 'tools/code_format/check_format.py fix'")
        sys.exit(1)

    if args.operation_type == "check":
        print("PASS")
