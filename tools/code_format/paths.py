#!/usr/bin/python3

import os
import os.path
import shutil

def get_buildifier():
    return shutil.which('buildifier')
