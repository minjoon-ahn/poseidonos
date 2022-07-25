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
import json_parser
import CREATE_TWO_RAID10_ARRAYS

ARRAY1NAME = CREATE_TWO_RAID10_ARRAYS.ARRAY1NAME
ARRAY2NAME = CREATE_TWO_RAID10_ARRAYS.ARRAY2NAME

def execute():
    CREATE_TWO_RAID10_ARRAYS.execute()
    isWT = True
    out1 = cli.mount_array(ARRAY1NAME, isWT)
    print(out1)
    code = json_parser.get_response_code(out1)
    if code is 0:
        isWT = False
        out2 = cli.mount_array(ARRAY2NAME, isWT)
        print(out2)
        return out2
    else:
        return out1

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        pos.set_addr(sys.argv[1])
    api.clear_result(__file__)
    out = execute()
    ret = api.set_result_by_code_eq(out, 0, __file__)
    pos.flush_and_kill_pos()
    exit(ret)