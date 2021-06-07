#!/usr/bin/env python3
import subprocess
import os
import sys
sys.path.append("../")
sys.path.append("../../system/lib/")

import json_parser
import pos
import pos_util
import cli
import api
import json
import STATE_BUSY_TO_STOP
import fio
import time

ARRAYNAME = STATE_BUSY_TO_STOP.ARRAYNAME

def execute():
    STATE_BUSY_TO_STOP.execute()
    out = cli.unmount_array(ARRAYNAME)
    return out

if __name__ == "__main__":
    api.clear_result(__file__)
    out = execute()
    ret = api.set_result_by_code_ne(out, 0, __file__)
    pos.kill_pos()
    exit(ret)