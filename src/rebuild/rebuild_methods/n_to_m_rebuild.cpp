/*
 *   BSD LICENSE
 *   Copyright (c) 2021 Samsung Electronics Corporation
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * Neither the name of Samsung Electronics Corporation nor the names of
 *       its contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <air/Air.h>

#include "n_to_m_rebuild.h"
#include "rebuild_read_done.h"
#include "rebuild_write_done.h"
#include "rebuild_partial_completion.h"
#include "src/bio/ubio.h"
#include "src/io_scheduler/io_dispatcher.h"
#include "src/logger/logger.h"

#include <functional>
using namespace std;

namespace pos
{
NToMRebuild::NToMRebuild(vector<IArrayDevice*> src, vector<IArrayDevice*> dst, RecoverFunc recoverFunc)
: RebuildMethod(src.size(), dst.size()),
  src(src),
  dst(dst),
  recoverFunc(recoverFunc)
{
    POS_TRACE_INFO(EID(REBUILD_DEBUG_MSG),
        "NToMRebuild constructor, srcCnt:{}, dstCnt:{}", src.size(), dst.size());
    airKey = (uint64_t)this;
}

NToMRebuild::~NToMRebuild()
{
    delete backupMethod;
    backupMethod = nullptr;
}

void
NToMRebuild::SetBackupMethod(NToMRebuild* backup)
{
    backupMethod = backup;
}

void
NToMRebuild::SetFailOver(void)
{
    if (backupMethod != nullptr)
    {
        isFailOver = true;
    }
}

bool NToMRebuild::IsFailOver(void)
{
    return isFailOver;
}

int
NToMRebuild::Recover(int arrayIndex, StripeId stripeId, const PartitionPhysicalSize* pSize, StripeRebuildDoneCallback callback)
{
    if (IsFailOver() == true)
    {
        return backupMethod->Recover(arrayIndex, stripeId, pSize, callback);
    }
    POS_TRACE_DEBUG(EID(REBUILD_DEBUG_MSG),
        "NToMRebuild, trying to read, array_idx:{}, stripe_id:{}", arrayIndex, stripeId);
    _Read(arrayIndex, stripeId, pSize, callback);
    return EID(SUCCESS);
}

void
NToMRebuild::_Read(int arrayIndex, StripeId stripeId, const PartitionPhysicalSize* pSize, StripeRebuildDoneCallback callback)
{
    airlog("LAT_SegmentRebuildRead", "begin", 0, airKey);
    uint32_t sectorCnt = srcSize / ArrayConfig::SECTOR_SIZE_BYTE;
    void* mem = nullptr;
    mem = srcBuffer->TryGetBuffer();
    if (nullptr == mem)
    {
        POS_TRACE_WARN(EID(RESOURCE_BUFFER_POOL_EMPTY),
            "Failed to get buffer during recoverchunk. {} Pool is empty", srcBuffer->GetOwner());
        // mem = Memory<ArrayConfig::SECTOR_SIZE_BYTE>::Alloc(sectorCnt);
    }
    assert(mem != nullptr);
    UbioSmartPtr readUbio(new Ubio(mem, sectorCnt, arrayIndex));

    vector<PhysicalBlkAddr> srcAddr;
    for (IArrayDevice* dev : src)
    {
        PhysicalBlkAddr pba;
        pba.arrayDev = dev;
        pba.lba = pSize->startLba +
            (stripeId * pSize->blksPerChunk) * ArrayConfig::SECTORS_PER_BLOCK;
        srcAddr.push_back(pba);
    }

    ReadDoneCallback readDoneCallback = bind(&NToMRebuild::_ReadDone, this,
        arrayIndex, stripeId, pSize, callback, mem, placeholders::_1);
    CallbackSmartPtr readCompletion(new RebuildReadDone(readUbio, readDoneCallback));
    readUbio->SetCallback(readCompletion);
    vector<UbioSmartPtr> splitList;
    for (auto pba : srcAddr)
    {
        bool isTail = false;
        UbioSmartPtr split =
            readUbio->Split(ChangeByteToSector(unitSize), isTail);
        split->SetPba(pba);
        CallbackSmartPtr event(
            new RebuildPartialCompletion(split));
        event->SetCallee(readCompletion);
        event->SetEventType(BackendEvent::BackendEvent_UserdataRebuild);
        split->SetEventType(BackendEvent::BackendEvent_UserdataRebuild);
        split->SetCallback(event);
        split->SetOriginUbio(readUbio);
        splitList.push_back(split);
    }

    readCompletion->SetWaitingCount(splitList.size());
    for (auto split : splitList)
    {
        IODispatcher* ioDispatcher = IODispatcherSingleton::Instance();
        ioDispatcher->Submit(split);
    }
}

void
NToMRebuild::_ReadDone(int arrayIndex, StripeId stripeId, const PartitionPhysicalSize* pSize, StripeRebuildDoneCallback callback, void* src, int result)
{
    airlog("LAT_SegmentRebuildRead", "end", 0, airKey);
    POS_TRACE_DEBUG(EID(REBUILD_DEBUG_MSG),
        "NToMRebuild, ReadDone, array_idx:{}, stripe_id:{}, result:{}", arrayIndex, stripeId, result);
    if (result != 0)
    {
        POS_TRACE_WARN(EID(REBUILD_DEBUG_MSG),
            "NToMRebuild, error occured during read, array_idx:{}, stripe_id:{}, result:{}", arrayIndex, stripeId, result);
        srcBuffer->ReturnBuffer(src);
        if (backupMethod != nullptr)
        {
            if (IsFailOver() == false)
            {
                bool ret = backupMethod->Init(owner + "_backup");
                if (ret == false)
                {
                    POS_TRACE_ERROR(EID(REBUILD_DEBUG_MSG), "failed to init backup method");
                    callback(result);
                }
            }
            SetFailOver();
            backupMethod->Recover(arrayIndex, stripeId, pSize, callback);
        }
        else
        {
            callback(result);
        }
        
        return;
    }
    _Recover(arrayIndex, stripeId, pSize, callback, src);
}

void
NToMRebuild::_Recover(int arrayIndex, StripeId stripeId, const PartitionPhysicalSize* pSize, StripeRebuildDoneCallback callback, void* src)
{
    airlog("LAT_SegmentRebuildRecover", "begin", 0, airKey);
    void* mem = nullptr;
    mem = dstBuffer->TryGetBuffer();
    if (nullptr == mem)
    {
        POS_TRACE_WARN(EID(RESOURCE_BUFFER_POOL_EMPTY),
            "Failed to get buffer during recover write. {} Pool is empty", dstBuffer->GetOwner());
        // uint32_t sectorCnt = dstSize / ArrayConfig::SECTOR_SIZE_BYTE;
        // mem = Memory<ArrayConfig::SECTOR_SIZE_BYTE>::Alloc(sectorCnt);
    }
    assert(mem != nullptr);
    recoverFunc(mem, src, dstSize);
    srcBuffer->ReturnBuffer(src);
    _RecoverDone(arrayIndex, stripeId, pSize, callback, mem, 0);
}

void
NToMRebuild::_RecoverDone(int arrayIndex, StripeId stripeId, const PartitionPhysicalSize* pSize, StripeRebuildDoneCallback callback, void* mem, int result)
{
    airlog("LAT_SegmentRebuildRecover", "end", 0, airKey);
    POS_TRACE_DEBUG(EID(REBUILD_DEBUG_MSG),
        "NToMRebuild, RecoverDone, array_idx:{}, stripe_id:{}, result:{}", arrayIndex, stripeId, result);
    if (result != 0)
    {
        POS_TRACE_WARN(EID(REBUILD_DEBUG_MSG),
            "NToMRebuild, error occured during recover, array_idx:{}, stripe_id:{}, result:{}", arrayIndex, stripeId, result);
        callback(result);
        return;
    }
    _Write(arrayIndex, stripeId, pSize, callback, mem);
}


void
NToMRebuild::_Write(int arrayIndex, StripeId stripeId, const PartitionPhysicalSize* pSize, StripeRebuildDoneCallback callback, void* mem)
{
    uint32_t sectorCnt = dstSize / ArrayConfig::SECTOR_SIZE_BYTE;
    airlog("LAT_SegmentRebuildWrite", "begin", 0, airKey);
    UbioSmartPtr writeUbio(new Ubio(mem, sectorCnt, arrayIndex));
    writeUbio->dir = UbioDir::Write;

    vector<PhysicalBlkAddr> dstAddr;
    for (IArrayDevice* dev : dst)
    {
        PhysicalBlkAddr pba;
        pba.arrayDev = dev;
        pba.lba = pSize->startLba +
            (stripeId * pSize->blksPerChunk) * ArrayConfig::SECTORS_PER_BLOCK;
        dstAddr.push_back(pba);
    }

    WriteDoneCallback writeDoneCallback = bind(&NToMRebuild::_WriteDone, this,
        arrayIndex, stripeId, callback, placeholders::_1);
    CallbackSmartPtr writeCompletion(new RebuildWriteDone(writeUbio, writeDoneCallback, dstBuffer));
    writeUbio->SetCallback(writeCompletion);
    vector<UbioSmartPtr> splitList;
    for (auto pba : dstAddr)
    {
        bool isTail = false;
        UbioSmartPtr split =
            writeUbio->Split(ChangeByteToSector(unitSize), isTail);
        split->SetPba(pba);
        CallbackSmartPtr event(
            new RebuildPartialCompletion(split));
        event->SetCallee(writeCompletion);
        event->SetEventType(BackendEvent::BackendEvent_UserdataRebuild);
        split->SetEventType(BackendEvent::BackendEvent_UserdataRebuild);
        split->SetCallback(event);
        split->SetOriginUbio(writeUbio);
        split->dir = UbioDir::Write;
        splitList.push_back(split);
    }

    writeCompletion->SetWaitingCount(splitList.size());
    for (auto split : splitList)
    {
        IODispatcher* ioDispatcher = IODispatcherSingleton::Instance();
        ioDispatcher->Submit(split);
    }
}

void
NToMRebuild::_WriteDone(int arrayIndex, StripeId stripeId, StripeRebuildDoneCallback callback, int result)
{
    airlog("LAT_SegmentRebuildWrite", "end", 0, airKey);
    POS_TRACE_DEBUG(EID(REBUILD_DEBUG_MSG),
        "NToMRebuild, writedone, array_idx:{}, stripe_id:{}, result:{}", arrayIndex, stripeId, result);
    if (result != 0)
    {
        POS_TRACE_WARN(EID(REBUILD_DEBUG_MSG),
            "NToMRebuild, error occured during write, array_idx:{}, stripe_id:{}, result:{}", arrayIndex, stripeId, result);
    }
    callback(result);
}

} // namespace pos
