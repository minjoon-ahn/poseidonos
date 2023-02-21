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

#include "pbr_loader.h"
#include "src/pbr/header/header_serializer.h"
#include "src/pbr/content/content_serializer_factory.h"
#include "src/pbr/io/pbr_reader.h"
#include "src/logger/logger.h"
#include <memory.h>
#include <rte_malloc.h>

namespace pbr
{
PbrLoader::PbrLoader(vector<pos::UblockSharedPtr> devs)
: PbrLoader(new HeaderSerializer(), new PbrReader(), devs)
{
}

PbrLoader::PbrLoader(IHeaderSerializer* headerSerializer,
    IPbrReader* pbrReader, vector<pos::UblockSharedPtr> devs)
: headerSerializer(headerSerializer),
  pbrReader(pbrReader),
  devs(devs)
{
}

PbrLoader::~PbrLoader(void)
{
    for (auto dev : devs)
    {
        dev = nullptr;
    }
    devs.clear();
    delete pbrReader;
    delete headerSerializer;
}

int
PbrLoader::Load(vector<AteData*>& ateListOut)
{
    int ret = 0;
    uint32_t pbrSize = header::TOTAL_PBR_SIZE;
    void* pbrData = rte_malloc(nullptr, pbrSize, pbrSize);
    for (auto dev : devs)
    {
        memset(pbrData, 0, pbrSize);
        int ret = pbrReader->Read(dev, (char*)pbrData, 0, pbrSize);
        if (ret == 0)
        {
            HeaderElement headerElem;
            ret = headerSerializer->Deserialize((char*)pbrData, header::LENGTH, &headerElem);
            if (ret == 0)
            {
                auto serializer = ContentSerializerFactory::GetSerializer(headerElem.revision);
                uint64_t startOffset = serializer->GetContentStartLba();
                AteData* ateData = nullptr;
                ret = serializer->Deserialize(ateData, &((char*)pbrData)[startOffset]);
                delete serializer;
                if (ret == 0)
                {
                    ateListOut.push_back(ateData);
                }
                else
                {
                    delete ateData;
                }
            }
        }
    }
    rte_free(pbrData);
    if (ateListOut.size() == 0)
    {
        ret = EID(PBR_LOAD_NO_VALID_PBR_FOUND);
    }
    return ret;
}
} // namespace pbr
