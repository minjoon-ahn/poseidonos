#!/usr/bin/env python3
import subprocess
import os
import sys
sys.path.append("../")
sys.path.append("../../system/lib/")
sys.path.append("../device_management/")

import json_parser
import pos
import cli
import api
import SCAN_DEV_BASIC

def execute():
    SCAN_DEV_BASIC.execute()
    cli.mbr_reset()
    out = cli.delete_array("POSArray")
    return out

if __name__ == "__main__":
    api.clear_result(__file__)
    out = execute()
    ret = api.set_result_by_code_ne(out, 0, __file__)
    pos.kill_pos()
    exit(ret)