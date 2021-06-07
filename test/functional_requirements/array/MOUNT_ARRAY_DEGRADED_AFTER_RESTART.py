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
import time
import MOUNT_ARRAY_NO_SPARE
ARRAYNAME = MOUNT_ARRAY_NO_SPARE.ARRAYNAME

def execute():
    MOUNT_ARRAY_NO_SPARE.execute()
    api.detach_ssd(MOUNT_ARRAY_NO_SPARE.DATA_DEV_1)
    cli.unmount_array(ARRAYNAME)
    pos.exit_pos()
    time.sleep(5)
    pos.start_pos()
    cli.scan_device()
    out = cli.mount_array(ARRAYNAME)
    return out

if __name__ == "__main__":
    api.clear_result(__file__)
    out = execute()
    ret = api.set_result_by_situation_eq(ARRAYNAME, out, "DEGRADED", __file__)
    pos.kill_pos()
    exit(ret)