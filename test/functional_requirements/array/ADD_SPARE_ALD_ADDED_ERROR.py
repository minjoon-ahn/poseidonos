#!/usr/bin/env python3
import subprocess
import os
import sys
sys.path.append("../")
sys.path.append("../../system/lib/")

import json_parser
import pos
import cli
import api
import json
import ADD_SPARE_BASIC

ARRAYNAME = ADD_SPARE_BASIC.ARRAYNAME

def execute():
    ADD_SPARE_BASIC.execute()
    out = cli.add_device(ADD_SPARE_BASIC.SPARE_DEV, ARRAYNAME)
    return out

if __name__ == "__main__":
    api.clear_result(__file__)
    out = execute()
    ret = api.set_result_by_code_ne(out, 0, __file__)
    pos.kill_pos()
    exit(ret)