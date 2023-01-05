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
#pragma once
#include <cassert>
#include <unistd.h>

#include "debug_info_maker.h"
#include "debug_info_queue.h"
namespace pos
{

template<typename T>
DebugInfoMaker<T>::DebugInfoMaker(void)
{
    run = false;
    timerUsec = DEFAULT_TIMER_VALUE;
    registered = false;
    infoName = "";
    debugInfoThread = new std::thread(&DebugInfoMaker<T>::_DebugInfoThread, this);
}

template<typename T>
DebugInfoMaker<T>::~DebugInfoMaker(void)
{
    run = false;
    if (nullptr != debugInfoThread)
    {
        debugInfoThread->join();
    }
    delete debugInfoThread;
    DeRegisterDebugInfo(infoName);
}

template<typename T>
void
DebugInfoMaker<T>::RegisterDebugInfo(std::string name, uint32_t entryCount, bool asyncLogging, uint64_t inputTimerUsec, bool enabled)
{
    if (inputTimerUsec != 0)
    {
        timerUsec = inputTimerUsec;
    }
    debugInfoQueue.RegisterDebugInfoQueue("History_" + name, entryCount, enabled);
    debugInfoQueueForError.RegisterDebugInfoQueue("History_" + name + "_Error", entryCount, enabled);
    debugInfoObject.RegisterDebugInfoInstance(name);
    if (asyncLogging)
    {
        run = true;
    }
    registered = true;
    infoName = name;
}

template<typename T>
void
DebugInfoMaker<T>::DeRegisterDebugInfo(std::string name)
{
    if (registered == true)
    {
        debugInfoQueue.DeRegisterDebugInfoQueue("History_" + name);
        debugInfoQueueForError.DeRegisterDebugInfoQueue("History_" + name + "_Error");
        debugInfoObject.DeRegisterDebugInfoInstance(name);
        registered = false;
    }
}

template<typename T>
DebugInfoOkay 
DebugInfoMaker<T>::IsOkay(T& obj)
{
    return DebugInfoOkay::PASS;
}

template<typename T>
void
DebugInfoMaker<T>::AddDebugInfo(uint64_t userSpecific)
{
    assert(registered == true);
    MakeDebugInfo(debugInfoObject);
    debugInfoObject.instanceOkay = IsOkay(debugInfoObject);
    if ((int)(debugInfoObject.summaryOkay) < (int)(debugInfoObject.instanceOkay))
    {
        debugInfoObject.summaryOkay = debugInfoObject.instanceOkay;
    }
    if (debugInfoObject.instanceOkay != DebugInfoOkay::PASS)
    {
        debugInfoQueueForError.AddDebugInfo(debugInfoObject, userSpecific);    
    }
    debugInfoQueue.AddDebugInfo(debugInfoObject, userSpecific);
}

template<typename T>
void
DebugInfoMaker<T>::SetTimer(uint64_t inputTimerUsec)
{
    timerUsec = inputTimerUsec;
}

template<typename T>
void
DebugInfoMaker<T>::_DebugInfoThread(void)
{
    cpu_set_t cpuSet = AffinityManagerSingleton::Instance()->GetCpuSet(CoreType::GENERAL_USAGE);
    sched_setaffinity(0, sizeof(cpuSet), &cpuSet);
    while(run)
    {
        AddDebugInfo(TIMER_TRIGGERED);
        usleep(timerUsec);
    }
}

}
