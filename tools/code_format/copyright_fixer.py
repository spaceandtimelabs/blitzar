#!/usr/bin/env python3

import sys
import re
import argparse
import pathlib
import sys
import datetime

SXT_COPYRIGHT_KEY = "Copyright"
SXT_COPYRIGHT_YEAR = SXT_COPYRIGHT_KEY + " XYZ"
SXT_COPYRIGHT_BEG_MARKER = "/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU."
SXT_COPYRIGHT_END_MARKER = " */"
SXT_COPYRIGHT = (
    """ *
 * """
    + SXT_COPYRIGHT_YEAR
    + """ Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License."""
)


def extract_copyright_year(block_comment):
    if len(block_comment) > 0:
        # Extract the current year from the existing block comment
        block_comment = "\n".join(block_comment)
        copyright_match = re.search(
            (SXT_COPYRIGHT_KEY + r'\s*\d+(\s*-\s*\d+|(\s*,\s*\d+)*)'), block_comment
        ).group(0)
        copyright_year = str(copyright_match).replace(SXT_COPYRIGHT_KEY, "").strip()

        return copyright_year
    else:
        # If there is no copyright, annotate the new copyright with the current year
        return str(datetime.date.today().year)


def read_copyright_block(all_lines):
    copyright_lines = []

    try:
        # Iterate through the lines until we find the end of the copyright block.
        while True:
            line = next(all_lines)
            if line == SXT_COPYRIGHT_END_MARKER:
                break
            copyright_lines.append(line)
    except StopIteration:
        print("Error: could not find end of copyright block", file=sys.stderr)
        sys.exit(1)

    return copyright_lines


# Fix the copyright header
def fix_copyright_header(path):
    pathl = pathlib.Path(path)
    source = pathl.read_text(encoding="utf-8")
    all_lines = iter(source.split("\n"))

    copyright_lines = []
    non_copyright_lines = []

    try:
        # Iterate through the lines until we find the beginning
        # of the copyright block or the end of the file.
        while True:
            line = next(all_lines)
            if line == SXT_COPYRIGHT_BEG_MARKER:
                copyright_lines = read_copyright_block(all_lines)
            else:
                non_copyright_lines.append(line)
    except StopIteration:
        pass

    # Collect the remaining lines in `non_copyright_lines`.
    non_copyright_lines += list(all_lines)

    copyright_year = extract_copyright_year(copyright_lines)

    # Update year only if the copyright notice doesn't exist.
    copyright_lines = str(SXT_COPYRIGHT).replace(
        SXT_COPYRIGHT_YEAR, SXT_COPYRIGHT_KEY + " " + copyright_year + "-present"
    )

    return "\n".join(
        filter(
            lambda x: x,
            [
                SXT_COPYRIGHT_BEG_MARKER,
                copyright_lines,
                SXT_COPYRIGHT_END_MARKER,
                "\n".join(non_copyright_lines),
            ],
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Copyright notice.")
    parser.add_argument("--path", type=str, help="Specify the path to the file")
    parser.add_argument(
        "--rewrite", action="store_true", help="Rewrite file in-place"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    copyrightd_source = fix_copyright_header(args.path)

    if args.rewrite:
        pathlib.Path(args.path).write_text(copyrightd_source, encoding="utf-8")
    else:
        sys.stdout.buffer.write(copyrightd_source.encode("utf-8"))
