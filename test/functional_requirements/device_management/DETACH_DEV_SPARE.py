#!/usr/bin/env python3
import subprocess
import os
import sys
sys.path.append("../")
sys.path.append("../../system/lib/")
sys.path.append("../volume/")
sys.path.append("../array/")

import json_parser
import pos
import pos_util
import cli
import api
import json
import time
import MOUNT_VOL_BASIC_1
import MOUNT_ARRAY_BASIC
DETACH_TARGET_DEV = MOUNT_ARRAY_BASIC.SPARE
ARRAYNAME = MOUNT_VOL_BASIC_1.ARRAYNAME

def check_result():
    if api.check_situation(ARRAYNAME, "NORMAL") == True:
        if api.is_device_exists(DETACH_TARGET_DEV) == False:
            return "pass"
    return "fail"

def execute():
    MOUNT_VOL_BASIC_1.execute()
    api.detach_ssd(DETACH_TARGET_DEV)
    time.sleep(0.1)

if __name__ == "__main__":
    api.clear_result(__file__)
    execute()
    result = check_result()
    ret = api.set_result_manually(cli.array_info(ARRAYNAME), result, __file__)
    pos.kill_pos()
    exit(ret)