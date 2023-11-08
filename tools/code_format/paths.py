#!/usr/bin/python3

import os
import os.path
import shutil

def get_buildifier():
    res = shutil.which('buildifier')
    if res:
        return res
    return "/usr/local/go/bin/buildifier"
