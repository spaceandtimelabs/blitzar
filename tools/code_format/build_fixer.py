#!/usr/bin/env python3

########################################
# Adopted from Envoy
#
# See third_party/license/envoy.LICENSE
########################################

# Enforces:
# - Misc. cleanups: avoids redundant blank lines, removes unused loads.

import subprocess
import sys
import paths

# Where does Buildifier live?
BUILDIFIER_PATH = paths.get_buildifier()

class SxtBuildFixerError(Exception): pass

# Run Buildifier commands on a string with lint mode.
def buildifier_lint(contents):
    r = subprocess.run([BUILDIFIER_PATH, '-lint=fix', '-mode=fix', '-type=build'],
                       input=contents.encode(),
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise SxtBuildFixerError('buildifier execution failed: %s' % r)

    return r.stdout.decode('utf-8')

def fix_build(path):
    with open(path, 'r') as f:
        contents = f.read()
    xforms = [
        buildifier_lint,
    ]
    for xform in xforms:
        contents = xform(contents)
    return contents

if __name__ == '__main__':
    if len(sys.argv) == 2:
        sys.stdout.write(fix_build(sys.argv[1]))
        sys.exit(0)
    elif len(sys.argv) == 3:
        reordered_source = fix_build(sys.argv[1])
        with open(sys.argv[2], 'w') as f:
            f.write(reordered_source)
        sys.exit(0)
    print('Usage: %s <source file path> [<destination file path>]' % sys.argv[0])
    sys.exit(1)
