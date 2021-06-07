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
import MOUNT_VOL_BASIC_1
import fio
import time
DETACH_TARGET_DEV = MOUNT_VOL_BASIC_1.ANY_DATA
ARRAYNAME = MOUNT_VOL_BASIC_1.ARRAYNAME

def execute():
    MOUNT_VOL_BASIC_1.execute()
    fio_proc = fio.start_fio(0, 120)
    time.sleep(10)
    api.detach_ssd(DETACH_TARGET_DEV)
    if api.wait_situation(ARRAYNAME, "REBUILDING") == True:
        if api.wait_situation(ARRAYNAME, "NORMAL") == True:
            fio.stop_fio(fio_proc)
            return "pass"
    fio.stop_fio(fio_proc)
    return "fail"

if __name__ == "__main__":
    api.clear_result(__file__)
    result = execute()
    ret = api.set_result_manually(cli.array_info(ARRAYNAME), result, __file__)
    pos.kill_pos()
    exit(ret)