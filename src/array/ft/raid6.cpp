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

#include "raid6.h"
#include "src/include/array_config.h"
#include "src/include/pos_event_id.h"
#include "src/array_models/dto/partition_physical_size.h"
#include "src/logger/logger.h"
#include "src/resource_manager/buffer_pool.h"
#include "src/helper/enumerable/query.h"

#include <string>
#include <isa-l.h>

namespace pos
{
Raid6::Raid6(const PartitionPhysicalSize* pSize, uint64_t bufferCntPerNuma, DevStateGetter devState)
: Method(RaidTypeEnum::RAID6),
  parityBufferCntPerNuma(bufferCntPerNuma)
{
    ftSize_ = {
        .minWriteBlkCnt = 0,
        .backupBlkCnt = pSize->blksPerChunk * 2,
        .blksPerChunk = pSize->blksPerChunk,
        .blksPerStripe = pSize->chunksPerStripe * pSize->blksPerChunk,
        .chunksPerStripe = pSize->chunksPerStripe};
    ftSize_.minWriteBlkCnt = ftSize_.blksPerStripe - ftSize_.backupBlkCnt;

    chunkCnt = ftSize_.chunksPerStripe;
    dataCnt = chunkCnt - parityCnt;
    chunkSize = ArrayConfig::BLOCK_SIZE_BYTE * ftSize_.blksPerChunk;
    stateGetter = devState;
    encode_matrix = new unsigned char[chunkCnt * dataCnt];
    gf_gen_cauchy1_matrix(encode_matrix, chunkCnt, dataCnt);
    g_tbls = new unsigned char[dataCnt * parityCnt * 32];
    ec_init_tables(dataCnt, parityCnt, &encode_matrix[dataCnt * dataCnt], g_tbls);
}

list<FtEntry>
Raid6::Translate(const LogicalEntry& le)
{
    vector<uint32_t> parityOffset = GetParityOffset(le.addr.stripeId);
    uint32_t offsetSizeforParity = ftSize_.blksPerChunk;
    assert(parityOffset.size() == 2);
    uint32_t pParityIndex = parityOffset.front();
    uint32_t qParityIndex = parityOffset.back();
    BlkOffset pParityOffset = (uint64_t)pParityIndex * (uint64_t)offsetSizeforParity;
    BlkOffset startOffset = le.addr.offset;
    BlkOffset lastOffset = startOffset + le.blkCnt - 1;
    uint32_t firstIndex = startOffset / ftSize_.blksPerChunk;
    uint32_t lastIndex = lastOffset / ftSize_.blksPerChunk;
    list<FtEntry> feList;
    FtEntry fe;
    fe.addr.stripeId = le.addr.stripeId;
    fe.addr.offset = le.addr.offset;
    fe.blkCnt = le.blkCnt;

    bool isPQSeparated = qParityIndex == 0;
    if (isPQSeparated)
    {
        uint32_t paritySize = ftSize_.blksPerChunk;
        fe.addr.offset += paritySize;
        feList.push_back(fe);
    }
    else
    {
        uint32_t paritySize = ftSize_.blksPerChunk * 2;
        if (pParityIndex <= firstIndex)
        {
            fe.addr.offset += paritySize;
            feList.push_back(fe);
        }
        else if (firstIndex < pParityIndex && pParityIndex <= lastIndex)
        {
            fe.blkCnt = pParityOffset - startOffset;
            feList.push_back(fe);
            FtEntry feSecond;
            feSecond.addr.stripeId = le.addr.stripeId;
            feSecond.addr.offset = pParityOffset + paritySize;
            feSecond.blkCnt = le.blkCnt - fe.blkCnt;
            feList.push_back(feSecond);
        }
        else
        {
            feList.push_back(fe);
        }
    }
    return feList;
}

vector<uint32_t>
Raid6::GetParityOffset(StripeId lsid)
{
    vector<uint32_t> raid6ParityIndex;

    uint32_t pParityOffset = lsid + dataCnt;
    uint32_t qParityOffset = pParityOffset + 1;

    uint32_t pParityIdx = pParityOffset % chunkCnt;
    uint32_t qParityIdx = qParityOffset % chunkCnt;

    raid6ParityIndex.push_back(pParityIdx);
    raid6ParityIndex.push_back(qParityIdx);

    return raid6ParityIndex;
}

int
Raid6::MakeParity(list<FtWriteEntry>& ftl, const LogicalWriteEntry& src)
{
    vector<uint32_t> parityOffset = GetParityOffset(src.addr.stripeId);
    assert(parityOffset.size() == parityCnt);

    pParityIndex = parityOffset.front();
    qParityIndex = parityOffset.back();

    FtWriteEntry fwe_pParity;
    FtWriteEntry fwe_qParity;

    fwe_pParity.addr.stripeId = src.addr.stripeId;
    fwe_pParity.addr.offset = (uint64_t)pParityIndex * (uint64_t)ftSize_.blksPerChunk;
    fwe_pParity.blkCnt = ftSize_.blksPerChunk;

    fwe_qParity.addr.stripeId = src.addr.stripeId;
    fwe_qParity.addr.offset = (uint64_t)qParityIndex * (uint64_t)ftSize_.blksPerChunk;
    fwe_qParity.blkCnt = ftSize_.blksPerChunk;

    BufferEntry pParity = _AllocChunk();
    BufferEntry qParity = _AllocChunk();
    list<BufferEntry> parities;

    parities.push_back(pParity);
    parities.push_back(qParity);

    _ComputePQParities(parities, *(src.buffers));
    assert(parities.size() == parityCnt);

    pParity = parities.front();
    qParity = parities.back();

    fwe_pParity.buffers.push_back(pParity);
    fwe_qParity.buffers.push_back(qParity);

    ftl.clear();
    ftl.push_back(fwe_pParity);
    ftl.push_back(fwe_qParity);

    return 0;
}

list<FtBlkAddr>
Raid6::GetRebuildGroup(FtBlkAddr fba, vector<ArrayDeviceState> devs)
{
    uint32_t  deviceStateIdx = 0;
    vector<int> abnormalIdx;

    for (auto devState : devs)
    {
        if (devState != ArrayDeviceState::NORMAL)
        {
            abnormalIdx.push_back(deviceStateIdx);
        }
        deviceStateIdx++;
    }

    uint32_t blksPerChunk = ftSize_.blksPerChunk;
    uint32_t offsetInChunk = fba.offset % blksPerChunk;
    uint32_t chunkIndex = fba.offset / blksPerChunk;

    list<FtBlkAddr> recoveryGroup;
    for (uint32_t i = 0; i < ftSize_.chunksPerStripe; i++)
    {
        if (i != chunkIndex && find(abnormalIdx.begin(), abnormalIdx.end(), i) == abnormalIdx.end())
        {
            FtBlkAddr fsa = {.stripeId = fba.stripeId,
                             .offset = offsetInChunk + i * blksPerChunk};
            recoveryGroup.push_back(fsa);
        }
    }
    return recoveryGroup;
}

RaidState
Raid6::GetRaidState(vector<ArrayDeviceState> devs)
{
    auto&& abnormalDevs = Enumerable::Where(devs,
        [](auto d) { return d != ArrayDeviceState::NORMAL; });

    POS_TRACE_INFO(EID(RAID_DEBUG_MSG), "GetRaidState from raid6, abnormal cnt:{} ", abnormalDevs.size());
    if (abnormalDevs.size() == 0)
    {
        return RaidState::NORMAL;
    }
    else if (abnormalDevs.size() <= 2)
    {
        return RaidState::DEGRADED;
    }
    return RaidState::FAILURE;
}

bool
Raid6::CheckNumofDevsToConfigure(uint32_t numofDevs)
{
    uint32_t minRequiredNumofDevsforRAID6 = 4;
    return numofDevs >= minRequiredNumofDevsforRAID6;
}

BufferEntry
Raid6::_AllocChunk()
{
    if (parityPools.size() == 0 && parityBufferCntPerNuma > 0)
    {
        POS_TRACE_WARN(EID(RAID_DEBUG_MSG),
            "Attempt to reallocate ParityPool because it is not allocated during creation, req_buffersPerNuma:{}",
            parityBufferCntPerNuma);
        bool ret = AllocParityPools(parityBufferCntPerNuma);
        if (ret == false)
        {
            int eventId = EID(CREATE_ARRAY_INSUFFICIENT_MEMORY_UNABLE_TO_ALLOC_PARITY_POOL);
            POS_TRACE_ERROR(eventId, "required number of buffers:{}", parityBufferCntPerNuma);
        }
    }

    uint32_t numa = affinityManager->GetNumaIdFromCurrentThread();
    BufferPool* bufferPool = parityPools.at(numa);
    void* mem = bufferPool->TryGetBuffer();

    // TODO error handling for the case of insufficient free parity buffer
    assert(nullptr != mem);

    BufferEntry buffer(mem, ftSize_.blksPerChunk, true);
    buffer.SetBufferPool(bufferPool);
    return buffer;
}

void
Raid6::_ComputePQParities(list<BufferEntry>& dst, const list<BufferEntry>& src)
{
    uint32_t src_iter, dst_iter;
    uint64_t* src_ptr = nullptr;
    uint64_t* dst_ptr = nullptr;
    unsigned char *sources[chunkCnt];

    src_iter = 0;

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        sources[i] = new unsigned char[chunkSize];
    }

    for (const BufferEntry& src_buffer : src)
    {
        src_ptr = (uint64_t*)src_buffer.GetBufferPtr();
        memcpy(sources[src_iter++], src_ptr, chunkSize);
    }

    ec_encode_data(chunkSize, dataCnt, parityCnt, g_tbls, sources, &sources[dataCnt]);

    dst_iter = 0;
    for (const BufferEntry& dst_buffer : dst)
    {
        dst_ptr = (uint64_t*)dst_buffer.GetBufferPtr();
        memcpy(dst_ptr, sources[dataCnt + dst_iter++], chunkSize);
    }

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        delete[] sources[i];
    }
}

void
Raid6::_RebuildData(void* dst, void* src, uint32_t dstSize, vector<uint32_t> readIndex)
{
    const vector<ArrayDeviceState> devs = stateGetter();
    uint32_t  deviceStateIdx = 0;
    vector<uint32_t> errorIndex;

    for (auto devState : devs)
    {
        if (devState != ArrayDeviceState::NORMAL)
        {
            errorIndex.push_back(deviceStateIdx);
        }
        deviceStateIdx++;
    }

    for(auto eIdx : errorIndex)
    {
        POS_TRACE_WARN(EID(RAID_DEBUG_MSG), "[1]Raid6::_RebuildData: err index:{}",eIdx);
    }

    uint32_t destCnt = errorIndex.size();
    assert(destCnt <= parityCnt);

    POS_TRACE_WARN(EID(RAID_DEBUG_MSG), "[2]Raid6::_RebuildData: destCnt:{}",destCnt);

    unsigned char err_index[chunkCnt];
    unsigned char decode_index[chunkCnt];
    unsigned char *recover_src[chunkCnt];
    unsigned char *recover_inp[chunkCnt];
    unsigned char *recover_outp[destCnt];
    unsigned char *temp_matrix = new unsigned char[chunkCnt * dataCnt];
    unsigned char *invert_matrix = new unsigned char[chunkCnt * dataCnt];
    unsigned char *decode_matrix = new unsigned char[chunkCnt * dataCnt];

    memset(err_index, 0, sizeof(err_index));
    
    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        recover_src[i] = new unsigned char[dstSize];
    }

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        recover_inp[i] = new unsigned char[dstSize];
    }

    for (uint32_t i = 0; i < destCnt; i++)
    {
        recover_outp[i] = new unsigned char[dstSize];
    }

    unsigned char* src_ptr = (unsigned char*)src;

    int idx = 0;
    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        if(find(errorIndex.begin(), errorIndex.end(),i) == errorIndex.end())
        {
            POS_TRACE_WARN(EID(RAID_DEBUG_MSG), "[3]Raid6::_RebuildData: recover source index:{}", i);
            memcpy(recover_src[i], src_ptr+(i-idx)*dstSize, dstSize);
        }
        else
        {
            memset(recover_src[i], 0, dstSize);
            idx++;
        }
    }

    for (uint32_t i = 0; i < destCnt; i++)
    {
        err_index[errorIndex[i]] = 1;
    }

    for (uint32_t i = 0, r = 0; i < dataCnt; i++, r++)
    {
        while (err_index[r])
        {
            r++;
        }
        for (uint32_t j = 0; j < dataCnt; j++)
        {
             temp_matrix[dataCnt * i + j] = encode_matrix[dataCnt * r + j];
        }
        decode_index[i] = r;
    }

    gf_invert_matrix(temp_matrix, invert_matrix, dataCnt);

    for (uint32_t i = 0; i < destCnt; i++)
    {
        if(errorIndex[i] == pParityIndex || errorIndex[i] == qParityIndex)
        {
            for (uint32_t pidx = 0; pidx < dataCnt; pidx++)
            {
                unsigned char s = 0;
                for (uint32_t j = 0; j < dataCnt; j++)
                {
                    s ^= gf_mul(invert_matrix[j * dataCnt + pidx], encode_matrix[dataCnt * errorIndex[i] + j]);
                }
                decode_matrix[dataCnt * i + pidx] = s;
            }
        }
        else
        {
            for (uint32_t j = 0; j < dataCnt; j++)
            {
                decode_matrix[dataCnt * i + j] = invert_matrix[dataCnt * errorIndex[i] + j];
            }
        }
    }

    for (uint32_t i = 0; i < dataCnt; i++)
    {
        recover_inp[i] = recover_src[decode_index[i]];
    }

    ec_init_tables(dataCnt, destCnt, decode_matrix, g_tbls);
    ec_encode_data(dstSize, dataCnt, destCnt, g_tbls, recover_inp, recover_outp);

    for (uint32_t i = 0; i < destCnt; i++)
    {
        if(find(readIndex.begin(), readIndex.end(), errorIndex[i]) != readIndex.end())
        {
            memcpy(dst, recover_outp[i], dstSize);
            POS_TRACE_WARN(EID(RAID_DEBUG_MSG), "[4]Raid6::_RebuildData[DONE]:{}",i);
        }
    }

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        delete[] recover_src[i];
    }

    for (uint32_t i = 0; i < destCnt; i++)
    {
        delete[] recover_outp[i];
    }

    delete[] decode_matrix;
    delete[] invert_matrix;
    delete[] temp_matrix;
}

bool
Raid6::AllocParityPools(uint64_t maxParityBufferCntPerNuma,
    AffinityManager* affMgr, MemoryManager* memoryMgr)
{
    affinityManager = affMgr;
    memoryManager = memoryMgr;
    const string NUMA_PREFIX = "_NUMA_";
    const uint64_t ARRAY_CHUNK_SIZE = ArrayConfig::BLOCK_SIZE_BYTE * ArrayConfig::BLOCKS_PER_CHUNK;

    uint32_t totalNumaCount = affinityManager->GetNumaCount();

    for (uint32_t numa = 0; numa < totalNumaCount; numa++)
    {
        BufferInfo info = {
            .owner = typeid(this).name() + NUMA_PREFIX + to_string(numa),
            .size = ARRAY_CHUNK_SIZE,
            .count = maxParityBufferCntPerNuma};
        BufferPool* pool = memoryManager->CreateBufferPool(info, numa);
        if (pool == nullptr)
        {
            ClearParityPools();
            return false;
        }
        parityPools.push_back(pool);
        POS_TRACE_DEBUG(EID(RAID_DEBUG_MSG), "BufferPool for RAID6 is created, {}", pool->GetOwner());
    }
    return true;
}

void
Raid6::ClearParityPools()
{
    for (unsigned int i = 0; i < parityPools.size(); i++)
    {
        if (parityPools[i] != nullptr)
        {
            POS_TRACE_DEBUG(EID(RAID_DEBUG_MSG), "ParityPool {} is cleared",
                parityPools[i]->GetOwner());
            memoryManager->DeleteBufferPool(parityPools[i]);
            parityPools[i] = nullptr;
        }
    }
    parityPools.clear();
}

Raid6::~Raid6()
{
    ClearParityPools();
    delete[] g_tbls;
    delete[] encode_matrix;
}

int
Raid6::GetParityPoolSize()
{
    return parityPools.size();
}

RecoverFunc
Raid6::GetRecoverFunc(int devIdx)
{
    vector<uint32_t> errorIndex;
    errorIndex.push_back(devIdx);
    recoverFunc = bind(&Raid6::_RebuildData, this, placeholders::_1, placeholders::_2, placeholders::_3, errorIndex);
    return recoverFunc;
}

} // namespace pos
