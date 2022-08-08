#include "src/array/ft/raid6.h"

#include <gtest/gtest.h>
#include <isa-l.h>
#include <typeinfo> 
#include <cstring>
#include <string.h>
#include "src/array_models/dto/partition_physical_size.h"
#include "src/include/array_config.h"
#include "src/include/array_device_state.h"
#include "test/unit-tests/cpu_affinity/affinity_manager_mock.h"
#include "test/unit-tests/resource_manager/buffer_pool_mock.h"
#include "test/unit-tests/resource_manager/memory_manager_mock.h"
#include "test/unit-tests/utils/mock_builder.h"

using ::testing::NiceMock;
using ::testing::Return;

typedef unsigned char BYTE;

namespace pos
{
static BufferEntry
generateRandomBufferEntry(int numBlocks, bool isParity)
{
    int bufSize = ArrayConfig::BLOCK_SIZE_BYTE * numBlocks;
    BYTE* buffer = new BYTE[bufSize];
    unsigned int seed = time(NULL);
    for (int i = 0; i < bufSize; i++)
    {
        //buffer[i] = rand_r(&seed);
        buffer[i] = 'A'+rand_r(&seed)%8;
    }
    return BufferEntry(buffer, numBlocks, isParity);
}

static BufferEntry
generateInitializedBufferEntry(int numBlocks, bool isParity)
{
    int bufSize = ArrayConfig::BLOCK_SIZE_BYTE * numBlocks;
    BYTE* buffer = new BYTE[bufSize]{0};
    return BufferEntry(buffer, numBlocks, isParity);
}

void
generateRSEncodingParities(BYTE** sources, BYTE* encode_matrix, int m, int k, int len)
{
    int i, j;
    int p = m - k;

    BYTE* g_tbls = new BYTE[k * p * 32];

    // Pick a Cauchy matrix as an encode matrix which is always invertible.
    gf_gen_cauchy1_matrix(
        encode_matrix, //[m x k] array to hold coefficients (32x30)
        m,             //number of rows in matrix corresponding to srcs + parity (32)
        k              //number of columns in matrix corresponding to srcs. (30)
    );

    // Init g_tables
    ec_init_tables(
        k,                     //The number of vector sources
        p,                     //Number of output vectors to concurrently encoding (p)
        &encode_matrix[k * k], //Pointer to sets of arrays of input coefficients used to encode
        g_tbls                 // Pointer to start of space for concatenated output tables generated from input coefficients. Must be of size 32∗k∗rows.
    );

    // Generate parities
    ec_encode_data(
        len,        //Length of each block of data (vector) of source or dest data.
        k,          //The number of vector sources or rows in the generator matrix for coding.
        p,          //The number of output vectors to concurrently encode/decode.
        g_tbls,     //Pointer to array of input tables generated from coding coefficients in ec_init_tables(). Must be of size 32∗k∗rows
        sources,    //Array of pointers to source input buffers
        &sources[k] //Array of pointers to coded output buffers.
    );

    delete[] g_tbls;
    return;
}

TEST(Raid6, Raid6_testPandQParityBufferGeneration)
{
    // Given
    const PartitionPhysicalSize physicalSize{
        .startLba = 0 /* not interesting */,
        .lastLba = 0 /* not interesting */,
        .blksPerChunk = 64,
        .chunksPerStripe = 4,
        .stripesPerSegment = 0 /* not interesting */,
        .totalSegments = 0 /* not interesting */
    };

    uint32_t i, j, iter = 0;
    uint32_t chunkCnt = physicalSize.chunksPerStripe;
    uint32_t parityCnt = 2;
    uint32_t dataCnt = chunkCnt - parityCnt;
    uint64_t* src_ptr = nullptr;
    uint32_t chunkSize = ArrayConfig::BLOCK_SIZE_BYTE * chunkCnt;

    MockAffinityManager mockAffMgr = BuildDefaultAffinityManagerMock();

    // When
    vector<ArrayDeviceState> vec;
    for(uint32_t i = 0; i < chunkCnt; i++)
    {
        vec.push_back(ArrayDeviceState::NORMAL);
    }

    Raid6 raid6(&physicalSize, 0, [&vec](){
        return vec;
    });

    std::list<BufferEntry> buffers;

    BYTE* testSources[chunkCnt];
    BYTE* encode_matrix = new BYTE[chunkCnt * dataCnt];

    for (i = 0; i < chunkCnt; i++)
    {
        testSources[i] = new BYTE[chunkSize];
    }

    int NUM_BLOCKS = physicalSize.blksPerChunk;

    BufferEntry be1 = generateRandomBufferEntry(NUM_BLOCKS, false);
    BufferEntry be2 = generateRandomBufferEntry(NUM_BLOCKS, false);

    buffers.push_back(be1);
    buffers.push_back(be2);

    for (const BufferEntry& src_buffer : buffers)
    {
        src_ptr = (uint64_t*)src_buffer.GetBufferPtr();
        memcpy(testSources[iter++], src_ptr, chunkSize);
    }

    generateRSEncodingParities(testSources, encode_matrix, chunkCnt, dataCnt, chunkSize);

    LogicalBlkAddr lBlkAddr{
        .stripeId = 0,
        .offset = 0};

    const LogicalWriteEntry& src{
        .addr = lBlkAddr,
        .blkCnt = physicalSize.blksPerChunk,
        .buffers = &buffers};

    list<BufferEntry> parities;

    BufferEntry pParity = generateInitializedBufferEntry(NUM_BLOCKS, true);
    BufferEntry qParity = generateInitializedBufferEntry(NUM_BLOCKS, true);

    parities.push_back(pParity);
    parities.push_back(qParity);

    raid6._ComputePQParities(parities, *(src.buffers));

    // Then
    ASSERT_EQ(2, parities.size());
    ASSERT_EQ(0, memcmp(testSources[dataCnt], parities.front().GetBufferPtr(), chunkSize));
    ASSERT_EQ(0, memcmp(testSources[dataCnt + 1], parities.back().GetBufferPtr(), chunkSize));

    for (i = 0; i < chunkCnt; i++)
    {
        delete[] testSources[i];
    }
    delete[] encode_matrix;
}

TEST(Raid6, Raid6_testGenerateSourceBuffersusingPQParityBuffers)
{
    // Given
    const PartitionPhysicalSize physicalSize{
        .startLba = 0 /* not interesting */,
        .lastLba = 0 /* not interesting */,
        .blksPerChunk = 64,
        .chunksPerStripe = 5,
        .stripesPerSegment = 0 /* not interesting */,
        .totalSegments = 0 /* not interesting */
    };

    uint32_t i, j, r, iter = 0;
    uint32_t chunkCnt = physicalSize.chunksPerStripe;
    uint32_t parityCnt = 2;
    uint32_t dataCnt = chunkCnt - parityCnt;
    uint64_t* src_ptr = nullptr;
    uint32_t chunkSize = ArrayConfig::BLOCK_SIZE_BYTE * chunkCnt;

    MockAffinityManager mockAffMgr = BuildDefaultAffinityManagerMock();

    // When
    vector<ArrayDeviceState> vec;
    for(uint32_t i = 0; i < chunkCnt; i++)
    {
            vec.push_back(ArrayDeviceState::NORMAL);
    }

    Raid6 raid6(&physicalSize, 0, [&vec](){
        return vec;

    });

    std::list<BufferEntry> buffers;

    BYTE* testSources[chunkCnt];
    BYTE* testOutput[chunkCnt];
    BYTE* encode_matrix = new BYTE[chunkCnt * dataCnt];

    for (i = 0; i < chunkCnt; i++)
    {
        testSources[i] = new BYTE[chunkSize];
        testOutput[i] = new BYTE[chunkSize];
    }

    int NUM_BLOCKS = physicalSize.blksPerChunk;

    BufferEntry be1 = generateRandomBufferEntry(NUM_BLOCKS, false);
    BufferEntry be2 = generateRandomBufferEntry(NUM_BLOCKS, false);
    BufferEntry be3 = generateRandomBufferEntry(NUM_BLOCKS, false);

    buffers.push_back(be1);
    buffers.push_back(be2);
    buffers.push_back(be3);

    for (const BufferEntry& src_buffer : buffers)
    {
        src_ptr = (uint64_t*)src_buffer.GetBufferPtr();
        memcpy(testSources[iter++], src_ptr, chunkSize);
    }

    LogicalBlkAddr lBlkAddr{
        .stripeId = 0,
        .offset = 0};

    const LogicalWriteEntry& src{
        .addr = lBlkAddr,
        .blkCnt = physicalSize.blksPerChunk,
        .buffers = &buffers};

    list<BufferEntry> parities;

    BufferEntry pParity = generateInitializedBufferEntry(NUM_BLOCKS, true);
    BufferEntry qParity = generateInitializedBufferEntry(NUM_BLOCKS, true);

    parities.push_back(pParity);
    parities.push_back(qParity);

    raid6._ComputePQParities(parities, *(src.buffers));

    ASSERT_EQ(2, parities.size());

    memcpy(testSources[dataCnt], parities.front().GetBufferPtr(), chunkSize);
    memcpy(testSources[dataCnt + 1], parities.back().GetBufferPtr(), chunkSize);

    // Then
    BYTE s;
    BYTE err_index[chunkCnt];
    BYTE decode_index[chunkCnt];
    BYTE* g_tbls = new BYTE[dataCnt * parityCnt * 32];
    BYTE* recover_srcs[dataCnt];
    BYTE err_list_for_test[parityCnt];
    BYTE* temp_matrix = new BYTE[chunkCnt * dataCnt];
    BYTE* invert_matrix = new BYTE[chunkCnt * dataCnt];
    BYTE* decode_matrix = new BYTE[chunkCnt * dataCnt];

    memset(err_index, 0, sizeof(err_index));
    gf_gen_cauchy1_matrix(encode_matrix, chunkCnt, dataCnt);
    // Select the index of losing buffers for simulation (be1, be3)
    err_list_for_test[0] = 0;
    err_list_for_test[1] = 2;

    // Order the fragments in erasure for easier sorting
    for (i = 0; i < parityCnt; i++)
    {
        err_index[err_list_for_test[i]] = 1;
    }

    // Construct b (matrix that encoded remaining frags) by removing erased rows
    for (i = 0, r = 0; i < dataCnt; i++, r++)
    {
        while (err_index[r])
            r++;
        for (j = 0; j < dataCnt; j++)
            temp_matrix[dataCnt * i + j] = encode_matrix[dataCnt * r + j];
        //r is the index of the survived buffers
        decode_index[i] = r;
    }

    gf_invert_matrix(temp_matrix, invert_matrix, dataCnt);

    // Get decode matrix with only wanted recovery rows
    for (i = 0; i < parityCnt; i++)
    {
        for (j = 0; j < dataCnt; j++)
            decode_matrix[dataCnt * i + j] = invert_matrix[dataCnt * err_list_for_test[i] + j];
    }

    for (i = 0; i < dataCnt; i++)
    {
        recover_srcs[i] = testSources[decode_index[i]];
    }

    ec_init_tables(dataCnt, parityCnt, decode_matrix, g_tbls);
    ec_encode_data(chunkSize, dataCnt, parityCnt, g_tbls, recover_srcs, testOutput);

    ASSERT_EQ(0, memcmp(testOutput[0], buffers.front().GetBufferPtr(), chunkSize));
    ASSERT_EQ(0, memcmp(testOutput[1], buffers.back().GetBufferPtr(), chunkSize));

    for (i = 0; i < chunkCnt; i++)
    {
        delete[] testSources[i];
        delete[] testOutput[i];
    }

    delete[] g_tbls;
    delete[] temp_matrix;
    delete[] invert_matrix;
    delete[] decode_matrix;
    delete[] encode_matrix;
}

// TEST(Raid6, Raid6_testIfTranslateCalculatesProperlyCaseWhereParityPAndParityQAreSeparated)
// {
//     // Given
//     uint32_t parityDevCnt = 2;
//     uint32_t dataDevCnt = 2;
//     uint32_t totalDevCnt = 4;
//     uint32_t blksPerChunk = 64;
 
//     const PartitionPhysicalSize physicalSize{
//         .startLba = 0 /* not interesting */,
//         .lastLba = 0 /* not interesting */,
//         .blksPerChunk = blksPerChunk,
//         .chunksPerStripe = totalDevCnt,
//         .stripesPerSegment = 0 /* not interesting */,
//         .totalSegments = 0 /* not interesting */};
//     uint32_t STRIPE_ID = 1; // Q D D P
//     uint32_t OFFSET = 0;
//     uint32_t expectedPLocation = (STRIPE_ID + totalDevCnt -2) % totalDevCnt;
//     uint32_t expectedQLocation = (STRIPE_ID + totalDevCnt -1) % totalDevCnt;

//     LogicalEntry src;
//     src.addr.stripeId = STRIPE_ID;
//     src.addr.offset = OFFSET;
//     src.blkCnt = dataDevCnt * blksPerChunk;

//     MockAffinityManager mockAffMgr = BuildDefaultAffinityManagerMock();
//     vector<ArrayDeviceState> vec;
//     for(uint32_t i = 0; i < totalDevCnt; i++)
//     {
//             vec.push_back(ArrayDeviceState::NORMAL);
//     }

//     Raid6 raid6(&physicalSize, 0, [&vec](){
//         return vec;
//     });

//     list<FtEntry> dest;

//     // When
//     dest = raid6.Translate(src);

//     // Then
//     ASSERT_EQ(1, dest.size());
//     FtEntry fte = dest.front();

//     ASSERT_EQ(STRIPE_ID, fte.addr.stripeId);
//     ASSERT_EQ(blksPerChunk, fte.addr.offset);
//     ASSERT_EQ(blksPerChunk * dataDevCnt, fte.blkCnt);
// }

TEST(Raid6, Raid6_testIfTranslateCalculatesProperlyCaseWhereParityPAndParityQAreAdjacent01)
{
    // Given
    uint32_t parityDevCnt = 2;
    uint32_t dataDevCnt = 2;
    uint32_t totalDevCnt = 4;
    uint32_t blksPerChunk = 64;

    const PartitionPhysicalSize physicalSize{
        .startLba = 0 /* not interesting */,
        .lastLba = 0 /* not interesting */,
        .blksPerChunk = blksPerChunk,
        .chunksPerStripe = totalDevCnt,
        .stripesPerSegment = 0 /* not interesting */,
        .totalSegments = 0 /* not interesting */};
    uint32_t STRIPE_ID = 0; //D D P Q 
    uint32_t OFFSET = 0;
    uint32_t expectedPLocation = (STRIPE_ID + totalDevCnt -2) % totalDevCnt;
    uint32_t expectedQLocation = (STRIPE_ID + totalDevCnt -1) % totalDevCnt;

    LogicalEntry src;
    src.addr.stripeId = STRIPE_ID;
    src.addr.offset = OFFSET;
    src.blkCnt = dataDevCnt * blksPerChunk;

    MockAffinityManager mockAffMgr = BuildDefaultAffinityManagerMock();
    vector<ArrayDeviceState> vec;
    for(uint32_t i = 0; i < totalDevCnt; i++)
    {
            vec.push_back(ArrayDeviceState::NORMAL);
    }

    Raid6 raid6(&physicalSize, 0, [&vec](){
        return vec;
    });

    list<FtEntry> dest;

    // When
    dest = raid6.Translate(src);

    // Then
    ASSERT_EQ(1, dest.size());
    FtEntry fte = dest.front();

    ASSERT_EQ(STRIPE_ID, fte.addr.stripeId);
    ASSERT_EQ(0, fte.addr.offset);
    ASSERT_EQ(blksPerChunk * dataDevCnt, fte.blkCnt);
}

TEST(Raid6, Raid6_testIfTranslateCalculatesProperlyCaseWhereParityPAndParityQAreAdjacent02)
{
    // Given
    uint32_t parityDevCnt = 2;
    uint32_t dataDevCnt = 2;
    uint32_t totalDevCnt = 4;
    uint32_t blksPerChunk = 64;

    const PartitionPhysicalSize physicalSize{
        .startLba = 0 /* not interesting */,
        .lastLba = 0 /* not interesting */,
        .blksPerChunk = blksPerChunk,
        .chunksPerStripe = totalDevCnt,
        .stripesPerSegment = 0 /* not interesting */,
        .totalSegments = 0 /* not interesting */};
    uint32_t STRIPE_ID = 2; //P Q D D
    uint32_t OFFSET = 0;
    uint32_t expectedPLocation = (STRIPE_ID + totalDevCnt -2) % totalDevCnt;
    uint32_t expectedQLocation = (STRIPE_ID + totalDevCnt -1) % totalDevCnt;

    LogicalEntry src;
    src.addr.stripeId = STRIPE_ID;
    src.addr.offset = OFFSET;
    src.blkCnt = dataDevCnt * blksPerChunk;

    MockAffinityManager mockAffMgr = BuildDefaultAffinityManagerMock();
    vector<ArrayDeviceState> vec;
    for(uint32_t i = 0; i < totalDevCnt; i++)
    {
            vec.push_back(ArrayDeviceState::NORMAL);
    }

    Raid6 raid6(&physicalSize, 0, [&vec](){
        return vec;
    });

    list<FtEntry> dest;

    // When
    dest = raid6.Translate(src);

    // Then
    ASSERT_EQ(1, dest.size());
    FtEntry fte = dest.front();

    ASSERT_EQ(STRIPE_ID, fte.addr.stripeId);
    ASSERT_EQ(blksPerChunk*parityDevCnt, fte.addr.offset);
    ASSERT_EQ(blksPerChunk * dataDevCnt, fte.blkCnt);
}

TEST(Raid6, Raid6_testIfTranslateCalculatesProperlyCaseWhereParityPAndParityQAreAdjacent03)
{
    // Given
    uint32_t parityDevCnt = 2;
    uint32_t dataDevCnt = 2;
    uint32_t totalDevCnt = 4;
    uint32_t blksPerChunk = 64;

    const PartitionPhysicalSize physicalSize{
        .startLba = 0 /* not interesting */,
        .lastLba = 0 /* not interesting */,
        .blksPerChunk = blksPerChunk,
        .chunksPerStripe = totalDevCnt,
        .stripesPerSegment = 0 /* not interesting */,
        .totalSegments = 0 /* not interesting */};
    uint32_t STRIPE_ID = 3;
    uint32_t OFFSET = 0;
    uint32_t expectedPLocation = (STRIPE_ID + totalDevCnt -2) % totalDevCnt;
    uint32_t expectedQLocation = (STRIPE_ID + totalDevCnt -1) % totalDevCnt;

    LogicalEntry src;
    src.addr.stripeId = STRIPE_ID;
    src.addr.offset = OFFSET;
    src.blkCnt = dataDevCnt * blksPerChunk;

    MockAffinityManager mockAffMgr = BuildDefaultAffinityManagerMock();
    vector<ArrayDeviceState> vec;
    for(uint32_t i = 0; i < totalDevCnt; i++)
    {
        vec.push_back(ArrayDeviceState::NORMAL);
    }

    Raid6 raid6(&physicalSize, 0, [&vec](){
        return vec;
    });

    list<FtEntry> dest;

    // When
    dest = raid6.Translate(src);

    // Then
    ASSERT_EQ(2, dest.size());
    FtEntry fteFront = dest.front();
    ASSERT_EQ(STRIPE_ID, fteFront.addr.stripeId);
    ASSERT_EQ(0, fteFront.addr.offset);
    ASSERT_EQ(blksPerChunk * expectedPLocation, fteFront.blkCnt);

    FtEntry fteBack = dest.back();
    ASSERT_EQ(STRIPE_ID, fteBack.addr.stripeId);
    ASSERT_EQ((expectedQLocation + 1) * blksPerChunk, fteBack.addr.offset);
    ASSERT_EQ(blksPerChunk * (dataDevCnt - 1), fteBack.blkCnt);
}
//////////////////////////////////////////////////////////////////////////////////////

TEST(Raid6, Raid6_testOneDeviceRebuildwithOneDataDeviceErrorCase01)
{
    // Given
    const PartitionPhysicalSize physicalSize{
        .startLba = 0 /* not interesting */,
        .lastLba = 0 /* not interesting */,
        .blksPerChunk = 4,
        .chunksPerStripe = 4,
        .stripesPerSegment = 0 /* not interesting */,
        .totalSegments = 0 /* not interesting */
    };

    // When
    uint32_t chunkCnt = physicalSize.chunksPerStripe; //# of total buffers
    uint32_t parityCnt = 2;                            //# of parity buffers
    uint32_t dataCnt = chunkCnt - parityCnt;                        //# of data buffers
    uint32_t chunkSize = physicalSize.blksPerChunk;
    
    vector<uint32_t> err_list_for_test;
    uint32_t firstErrIdx = 0;
    //uint32_t secondErrIdx = 1;

    err_list_for_test.push_back(firstErrIdx);
    //err_list_for_test.push_back(secondErrIdx);
    uint32_t nerrs = err_list_for_test.size();

    BYTE* encoding_sources[chunkCnt];
    BYTE* encoding_results_with_errors;
    BYTE* decoding_results;

    BYTE* encode_matrix = new BYTE[chunkCnt * dataCnt];
    BYTE* g_tbls = new BYTE[dataCnt * parityCnt * 32];

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        encoding_sources[i] = new BYTE[chunkSize];
    }

    encoding_results_with_errors = new BYTE[chunkSize * (chunkCnt - nerrs)];

    decoding_results = new BYTE[chunkSize * nerrs];

    unsigned int seed = time(NULL);
    for(uint32_t i = 0; i <dataCnt; i++)
    {
         for(uint32_t j = 0; j < chunkSize; j++)
        {
            encoding_sources[i][j] = rand_r(&seed);
        }
    }

    // for(uint32_t i = 0; i <dataCnt; i++)
    // {
    //      for(uint32_t j = 0; j < chunkSize; j++)
    //     {
    //         encoding_sources[i][j] = 'A'+i+j;
    //     }
    // }

    gf_gen_cauchy1_matrix(encode_matrix, chunkCnt, dataCnt);
    ec_init_tables(dataCnt, parityCnt, &encode_matrix[dataCnt * dataCnt], g_tbls);
    ec_encode_data(chunkSize, dataCnt, parityCnt, g_tbls, encoding_sources, &encoding_sources[dataCnt]);

    // cout<<"encoding_results_with_parities"<<endl;
    // for (uint32_t i = 0; i < chunkCnt; i++)
    // {
    //     for(uint32_t j = 0; j < chunkSize; j++)
    //     {
    //         printf("[%c]", encoding_sources[i][j]);
    //     }
    //     cout <<endl;
    // }

    vector<ArrayDeviceState> vec;
    for(uint32_t i = 0; i < chunkCnt; i++)
    {
        if(i == firstErrIdx)
        {
            vec.push_back(ArrayDeviceState::FAULT);
        }
        else
        {
            vec.push_back(ArrayDeviceState::NORMAL);
        }
    }

    Raid6 raid6(&physicalSize, 0, [&vec](){
        return vec;
    });
    uint32_t tempInx = 0;
    vector<uint32_t> idx_for_test;
    for(uint32_t i = 0; i<chunkCnt; i++)
    {
        if(find(err_list_for_test.begin(), err_list_for_test.end(), i) == err_list_for_test.end())
        {
            idx_for_test.push_back(tempInx);
        }
        tempInx++;
    }

    uint32_t src_i = 0;
    for(auto idx:idx_for_test)
    {
        for(uint32_t j =0; j < chunkSize; j++)
        {
            encoding_results_with_errors[src_i*chunkSize + j] = encoding_sources[idx][j];
        }
        src_i++;
    }

    // cout<<"encoding_results_with_errors"<<endl;
    // for (uint32_t i = 0; i < (chunkCnt-nerrs)*chunkSize; i++)
    // {
    //     printf("[%c]",encoding_results_with_errors[i]);
    //       if (i == chunkSize -1)
    //       {
    //         cout <<endl;
    //       }
    //     cout <<endl;
    // }

    //err_list_for_test.pop_back();
    raid6._RebuildData(decoding_results, encoding_results_with_errors, chunkSize, err_list_for_test, 0);

    // cout<<"decoding_results"<<endl;
    // for (uint32_t i = 0; i < err_list_for_test.size()*chunkSize; i++)
    // {
    //     printf("[%c]", decoding_results[i]);
    //     if (i == chunkCnt -1)
    //         cout <<endl;
    // }

    for(uint32_t i = 0; i < err_list_for_test.size(); i++)
    {
        ASSERT_EQ(0, memcmp(decoding_results, encoding_sources[err_list_for_test[i]], chunkSize));
    }

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        delete[] encoding_sources[i];
    }

    delete encoding_results_with_errors;
     
    delete decoding_results;

    delete[] g_tbls;
    delete[] encode_matrix;
}

TEST(Raid6, Raid6_testOneDeviceRebuildwithOneDataDeviceErrorCase02)
{
    // Given
    const PartitionPhysicalSize physicalSize{
        .startLba = 0 /* not interesting */,
        .lastLba = 0 /* not interesting */,
        .blksPerChunk = 4,
        .chunksPerStripe = 4,
        .stripesPerSegment = 0 /* not interesting */,
        .totalSegments = 0 /* not interesting */
    };

    // When
    uint32_t chunkCnt = physicalSize.chunksPerStripe; //# of total buffers
    uint32_t parityCnt = 2;                            //# of parity buffers
    uint32_t dataCnt = chunkCnt - parityCnt;                        //# of data buffers
    uint32_t chunkSize = physicalSize.blksPerChunk;
    
    vector<uint32_t> err_list_for_test;
    uint32_t firstErrIdx = 1;
    //uint32_t secondErrIdx = 1;

    err_list_for_test.push_back(firstErrIdx);
    //err_list_for_test.push_back(secondErrIdx);
    uint32_t nerrs = err_list_for_test.size();

    BYTE* encoding_sources[chunkCnt];
    BYTE* encoding_results_with_errors;
    BYTE* decoding_results;

    BYTE* encode_matrix = new BYTE[chunkCnt * dataCnt];
    BYTE* g_tbls = new BYTE[dataCnt * parityCnt * 32];

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        encoding_sources[i] = new BYTE[chunkSize];
    }

    encoding_results_with_errors = new BYTE[chunkSize * (chunkCnt - nerrs)];

    decoding_results = new BYTE[chunkSize * nerrs];

    unsigned int seed = time(NULL);
    for(uint32_t i = 0; i <dataCnt; i++)
    {
         for(uint32_t j = 0; j < chunkSize; j++)
        {
            encoding_sources[i][j] = rand_r(&seed);
        }
    }

    // for(uint32_t i = 0; i <dataCnt; i++)
    // {
    //      for(uint32_t j = 0; j < chunkSize; j++)
    //     {
    //         encoding_sources[i][j] = 'A'+i+j;
    //     }
    // }

    gf_gen_cauchy1_matrix(encode_matrix, chunkCnt, dataCnt);
    ec_init_tables(dataCnt, parityCnt, &encode_matrix[dataCnt * dataCnt], g_tbls);
    ec_encode_data(chunkSize, dataCnt, parityCnt, g_tbls, encoding_sources, &encoding_sources[dataCnt]);

    // cout<<"encoding_results_with_parities"<<endl;
    // for (uint32_t i = 0; i < chunkCnt; i++)
    // {
    //     for(uint32_t j = 0; j < chunkSize; j++)
    //     {
    //         printf("[%c]", encoding_sources[i][j]);
    //     }
    //     cout <<endl;
    // }

    vector<ArrayDeviceState> vec;
    for(uint32_t i = 0; i < chunkCnt; i++)
    {
        if(i == firstErrIdx)
        {
            vec.push_back(ArrayDeviceState::FAULT);
        }
        else
        {
            vec.push_back(ArrayDeviceState::NORMAL);
        }
    }

    Raid6 raid6(&physicalSize, 0, [&vec](){
        return vec;
    });
    uint32_t tempInx = 0;
    vector<uint32_t> idx_for_test;
    for(uint32_t i = 0; i<chunkCnt; i++)
    {
        if(find(err_list_for_test.begin(), err_list_for_test.end(), i) == err_list_for_test.end())
        {
            idx_for_test.push_back(tempInx);
        }
        tempInx++;
    }

    uint32_t src_i = 0;
    for(auto idx:idx_for_test)
    {
        for(uint32_t j =0; j < chunkSize; j++)
        {
            encoding_results_with_errors[src_i*chunkSize + j] = encoding_sources[idx][j];
        }
        src_i++;
    }

    // cout<<"encoding_results_with_errors"<<endl;
    // for (uint32_t i = 0; i < (chunkCnt-nerrs)*chunkSize; i++)
    // {
    //     printf("[%c]",encoding_results_with_errors[i]);
    //       if (i == chunkSize -1)
    //       {
    //         cout <<endl;
    //       }
    //     cout <<endl;
    // }

    //err_list_for_test.pop_back();
    raid6._RebuildData(decoding_results, encoding_results_with_errors, chunkSize, err_list_for_test, 0);

    // cout<<"decoding_results"<<endl;
    // for (uint32_t i = 0; i < err_list_for_test.size()*chunkSize; i++)
    // {
    //     printf("[%c]", decoding_results[i]);
    //     if (i == chunkCnt -1)
    //         cout <<endl;
    // }

    for(uint32_t i = 0; i < err_list_for_test.size(); i++)
    {
        ASSERT_EQ(0, memcmp(decoding_results, encoding_sources[err_list_for_test[i]], chunkSize));
    }

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        delete[] encoding_sources[i];
    }

    delete encoding_results_with_errors;
     
    delete decoding_results;

    delete[] g_tbls;
    delete[] encode_matrix;
}

TEST(Raid6, Raid6_testOneDeviceRebuildwithOneParityDeviceError01)
{
    // Given
    const PartitionPhysicalSize physicalSize{
        .startLba = 0 /* not interesting */,
        .lastLba = 0 /* not interesting */,
        .blksPerChunk = 4,
        .chunksPerStripe = 4,
        .stripesPerSegment = 0 /* not interesting */,
        .totalSegments = 0 /* not interesting */
    };

    // When
    uint32_t chunkCnt = physicalSize.chunksPerStripe; //# of total buffers
    uint32_t parityCnt = 2;                            //# of parity buffers
    uint32_t dataCnt = chunkCnt - parityCnt;                        //# of data buffers
    uint32_t chunkSize = physicalSize.blksPerChunk;
    
    vector<uint32_t> err_list_for_test;
    uint32_t firstErrIdx = chunkCnt-1;
    //uint32_t secondErrIdx = 1;

    err_list_for_test.push_back(firstErrIdx);
    //err_list_for_test.push_back(secondErrIdx);
    uint32_t nerrs = err_list_for_test.size();

    BYTE* encoding_sources[chunkCnt];
    BYTE* encoding_results_with_errors;
    BYTE* decoding_results;

    BYTE* encode_matrix = new BYTE[chunkCnt * dataCnt];
    BYTE* g_tbls = new BYTE[dataCnt * parityCnt * 32];

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        encoding_sources[i] = new BYTE[chunkSize];
    }

    encoding_results_with_errors = new BYTE[chunkSize * (chunkCnt - nerrs)];

    decoding_results = new BYTE[chunkSize * nerrs];

    unsigned int seed = time(NULL);
    for(uint32_t i = 0; i <dataCnt; i++)
    {
         for(uint32_t j = 0; j < chunkSize; j++)
        {
            encoding_sources[i][j] = rand_r(&seed);
        }
    }

    gf_gen_cauchy1_matrix(encode_matrix, chunkCnt, dataCnt);
    ec_init_tables(dataCnt, parityCnt, &encode_matrix[dataCnt * dataCnt], g_tbls);
    ec_encode_data(chunkSize, dataCnt, parityCnt, g_tbls, encoding_sources, &encoding_sources[dataCnt]);

    vector<ArrayDeviceState> vec;
    for(uint32_t i = 0; i < chunkCnt; i++)
    {
        if(i == firstErrIdx)
        {
            vec.push_back(ArrayDeviceState::FAULT);
        }
        else
        {
            vec.push_back(ArrayDeviceState::NORMAL);
        }
    }

    Raid6 raid6(&physicalSize, 0, [&vec](){
        return vec;
    });
    uint32_t tempInx = 0;
    vector<uint32_t> idx_for_test;
    for(uint32_t i = 0; i<chunkCnt; i++)
    {
        if(find(err_list_for_test.begin(), err_list_for_test.end(), i) == err_list_for_test.end())
        {
            idx_for_test.push_back(tempInx);
        }
        tempInx++;
    }

    uint32_t src_i = 0;
    for(auto idx:idx_for_test)
    {
        for(uint32_t j =0; j < chunkSize; j++)
        {
            encoding_results_with_errors[src_i*chunkSize + j] = encoding_sources[idx][j];
        }
        src_i++;
    }
    //err_list_for_test.pop_back();
    raid6._RebuildData(decoding_results, encoding_results_with_errors, chunkSize, err_list_for_test,0);

    for(uint32_t i = 0; i < err_list_for_test.size(); i++)
    {
        ASSERT_EQ(0, memcmp(decoding_results, encoding_sources[err_list_for_test[i]], chunkSize));
    }

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        delete[] encoding_sources[i];
    }

    delete encoding_results_with_errors;
     
    delete decoding_results;

    delete[] g_tbls;
    delete[] encode_matrix;
}

TEST(Raid6, Raid6_testOneDeviceRebuildwithOneParityDeviceError02)
{
    // Given
    const PartitionPhysicalSize physicalSize{
        .startLba = 0 /* not interesting */,
        .lastLba = 0 /* not interesting */,
        .blksPerChunk = 4,
        .chunksPerStripe = 4,
        .stripesPerSegment = 0 /* not interesting */,
        .totalSegments = 0 /* not interesting */
    };

    // When
    uint32_t chunkCnt = physicalSize.chunksPerStripe; //# of total buffers
    uint32_t parityCnt = 2;                            //# of parity buffers
    uint32_t dataCnt = chunkCnt - parityCnt;                        //# of data buffers
    uint32_t chunkSize = physicalSize.blksPerChunk;
    
    vector<uint32_t> err_list_for_test;
    uint32_t firstErrIdx = chunkCnt-2;
    //uint32_t secondErrIdx = 1;

    err_list_for_test.push_back(firstErrIdx);
    //err_list_for_test.push_back(secondErrIdx);
    uint32_t nerrs = err_list_for_test.size();

    BYTE* encoding_sources[chunkCnt];
    BYTE* encoding_results_with_errors;
    BYTE* decoding_results;

    BYTE* encode_matrix = new BYTE[chunkCnt * dataCnt];
    BYTE* g_tbls = new BYTE[dataCnt * parityCnt * 32];

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        encoding_sources[i] = new BYTE[chunkSize];
    }

    encoding_results_with_errors = new BYTE[chunkSize * (chunkCnt - nerrs)];

    decoding_results = new BYTE[chunkSize * nerrs];

    unsigned int seed = time(NULL);
    for(uint32_t i = 0; i <dataCnt; i++)
    {
         for(uint32_t j = 0; j < chunkSize; j++)
        {
            encoding_sources[i][j] = rand_r(&seed);
        }
    }

    gf_gen_cauchy1_matrix(encode_matrix, chunkCnt, dataCnt);
    ec_init_tables(dataCnt, parityCnt, &encode_matrix[dataCnt * dataCnt], g_tbls);
    ec_encode_data(chunkSize, dataCnt, parityCnt, g_tbls, encoding_sources, &encoding_sources[dataCnt]);

    vector<ArrayDeviceState> vec;
    for(uint32_t i = 0; i < chunkCnt; i++)
    {
        if(i == firstErrIdx)
        {
            vec.push_back(ArrayDeviceState::FAULT);
        }
        else
        {
            vec.push_back(ArrayDeviceState::NORMAL);
        }
    }

    Raid6 raid6(&physicalSize, 0, [&vec](){
        return vec;
    });
    uint32_t tempInx = 0;
    vector<uint32_t> idx_for_test;
    for(uint32_t i = 0; i<chunkCnt; i++)
    {
        if(find(err_list_for_test.begin(), err_list_for_test.end(), i) == err_list_for_test.end())
        {
            idx_for_test.push_back(tempInx);
        }
        tempInx++;
    }

    uint32_t src_i = 0;
    for(auto idx:idx_for_test)
    {
        for(uint32_t j =0; j < chunkSize; j++)
        {
            encoding_results_with_errors[src_i*chunkSize + j] = encoding_sources[idx][j];
        }
        src_i++;
    }
    //err_list_for_test.pop_back();
    raid6._RebuildData(decoding_results, encoding_results_with_errors, chunkSize, err_list_for_test,0);

    for(uint32_t i = 0; i < err_list_for_test.size(); i++)
    {
        ASSERT_EQ(0, memcmp(decoding_results, encoding_sources[err_list_for_test[i]], chunkSize));
    }

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        delete[] encoding_sources[i];
    }

    delete encoding_results_with_errors;
     
    delete decoding_results;

    delete[] g_tbls;
    delete[] encode_matrix;
}

TEST(Raid6, Raid6_testOneDeviceRebuildwithTwoDataDeviceErrors01)
{
    // Given
    const PartitionPhysicalSize physicalSize{
        .startLba = 0 /* not interesting */,
        .lastLba = 0 /* not interesting */,
        .blksPerChunk = 64,
        .chunksPerStripe = 12,
        .stripesPerSegment = 0 /* not interesting */,
        .totalSegments = 0 /* not interesting */
    };

    // When
    uint32_t chunkCnt = physicalSize.chunksPerStripe; //# of total buffers
    uint32_t parityCnt = 2;                            //# of parity buffers
    uint32_t dataCnt = chunkCnt - parityCnt;                        //# of data buffers
    uint32_t chunkSize = physicalSize.blksPerChunk;
    
    vector<uint32_t> err_list_for_test;
    uint32_t firstErrIdx = 0;
    uint32_t secondErrIdx = 1;

    err_list_for_test.push_back(firstErrIdx);
    err_list_for_test.push_back(secondErrIdx);
    uint32_t nerrs = err_list_for_test.size();

    BYTE* encoding_sources[chunkCnt];
    BYTE* encoding_results_with_errors;
    BYTE* decoding_results;

    BYTE* encode_matrix = new BYTE[chunkCnt * dataCnt];
    BYTE* g_tbls = new BYTE[dataCnt * parityCnt * 32];

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        encoding_sources[i] = new BYTE[chunkSize];
    }

    encoding_results_with_errors = new BYTE[chunkSize * (chunkCnt - nerrs)];

    decoding_results = new BYTE[chunkSize * nerrs];

    unsigned int seed = time(NULL);
    for(uint32_t i = 0; i <dataCnt; i++)
    {
         for(uint32_t j = 0; j < chunkSize; j++)
        {
            encoding_sources[i][j] = rand_r(&seed);
        }
    }

    gf_gen_cauchy1_matrix(encode_matrix, chunkCnt, dataCnt);
    ec_init_tables(dataCnt, parityCnt, &encode_matrix[dataCnt * dataCnt], g_tbls);
    ec_encode_data(chunkSize, dataCnt, parityCnt, g_tbls, encoding_sources, &encoding_sources[dataCnt]);

    vector<ArrayDeviceState> vec;
    for(uint32_t i = 0; i < chunkCnt; i++)
    {
        if(i == firstErrIdx || i == secondErrIdx )
        {
            vec.push_back(ArrayDeviceState::FAULT);
        }
        else
        {
            vec.push_back(ArrayDeviceState::NORMAL);
        }

    }

    Raid6 raid6(&physicalSize, 0, [&vec](){
        return vec;
    });
    uint32_t tempInx = 0;
    vector<uint32_t> idx_for_test;
    for(uint32_t i = 0; i<chunkCnt; i++)
    {
        if(find(err_list_for_test.begin(), err_list_for_test.end(), i) == err_list_for_test.end())
        {
            idx_for_test.push_back(tempInx);
        }
        tempInx++;
    }

    uint32_t src_i = 0;
    for(auto idx:idx_for_test)
    {
        for(uint32_t j =0; j < chunkSize; j++)
        {
            encoding_results_with_errors[src_i*chunkSize + j] = encoding_sources[idx][j];
        }
        src_i++;
    }

    err_list_for_test.clear();
    err_list_for_test.push_back(firstErrIdx);
    raid6._RebuildData(decoding_results, encoding_results_with_errors, chunkSize, err_list_for_test, 0);

    for(uint32_t i = 0; i < err_list_for_test.size(); i++)
    {
        ASSERT_EQ(0, memcmp(decoding_results, encoding_sources[err_list_for_test[i]], chunkSize));
    }

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        delete[] encoding_sources[i];
    }

    delete encoding_results_with_errors;
     
    delete decoding_results;

    delete[] g_tbls;
    delete[] encode_matrix;
}


TEST(Raid6, Raid6_testOneDeviceRebuildwithTwoDataDeviceErrorsCase02)
{
    // Given
    const PartitionPhysicalSize physicalSize{
        .startLba = 0 /* not interesting */,
        .lastLba = 0 /* not interesting */,
        .blksPerChunk = 64,
        .chunksPerStripe = 12,
        .stripesPerSegment = 0 /* not interesting */,
        .totalSegments = 0 /* not interesting */
    };

    // When
    uint32_t chunkCnt = physicalSize.chunksPerStripe; //# of total buffers
    uint32_t parityCnt = 2;                            //# of parity buffers
    uint32_t dataCnt = chunkCnt - parityCnt;                        //# of data buffers
    uint32_t chunkSize = physicalSize.blksPerChunk;
    
    vector<uint32_t> err_list_for_test;
    uint32_t firstErrIdx = 0;
    uint32_t secondErrIdx = 1;

    err_list_for_test.push_back(firstErrIdx);
    err_list_for_test.push_back(secondErrIdx);
    uint32_t nerrs = err_list_for_test.size();

    BYTE* encoding_sources[chunkCnt];
    BYTE* encoding_results_with_errors;
    BYTE* decoding_results;

    BYTE* encode_matrix = new BYTE[chunkCnt * dataCnt];
    BYTE* g_tbls = new BYTE[dataCnt * parityCnt * 32];

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        encoding_sources[i] = new BYTE[chunkSize];
    }

    encoding_results_with_errors = new BYTE[chunkSize * (chunkCnt - nerrs)];

    decoding_results = new BYTE[chunkSize * nerrs];

    unsigned int seed = time(NULL);
    for(uint32_t i = 0; i <dataCnt; i++)
    {
         for(uint32_t j = 0; j < chunkSize; j++)
        {
            encoding_sources[i][j] = rand_r(&seed);
        }
    }

    gf_gen_cauchy1_matrix(encode_matrix, chunkCnt, dataCnt);
    ec_init_tables(dataCnt, parityCnt, &encode_matrix[dataCnt * dataCnt], g_tbls);
    ec_encode_data(chunkSize, dataCnt, parityCnt, g_tbls, encoding_sources, &encoding_sources[dataCnt]);

    vector<ArrayDeviceState> vec;
    for(uint32_t i = 0; i < chunkCnt; i++)
    {
        if(i == firstErrIdx || i == secondErrIdx )
        {
            vec.push_back(ArrayDeviceState::FAULT);
        }
        else
        {
            vec.push_back(ArrayDeviceState::NORMAL);
        }

    }

    Raid6 raid6(&physicalSize, 0, [&vec](){
        return vec;
    });
    uint32_t tempInx = 0;
    vector<uint32_t> idx_for_test;
    for(uint32_t i = 0; i<chunkCnt; i++)
    {
        if(find(err_list_for_test.begin(), err_list_for_test.end(), i) == err_list_for_test.end())
        {
            idx_for_test.push_back(tempInx);
        }
        tempInx++;
    }

    uint32_t src_i = 0;
    for(auto idx:idx_for_test)
    {
        for(uint32_t j =0; j < chunkSize; j++)
        {
            encoding_results_with_errors[src_i*chunkSize + j] = encoding_sources[idx][j];
        }
        src_i++;
    }

    err_list_for_test.clear();
    err_list_for_test.push_back(secondErrIdx);
    raid6._RebuildData(decoding_results, encoding_results_with_errors, chunkSize, err_list_for_test,0);

    for(uint32_t i = 0; i < err_list_for_test.size(); i++)
    {
        ASSERT_EQ(0, memcmp(decoding_results, encoding_sources[err_list_for_test[i]], chunkSize));
    }

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        delete[] encoding_sources[i];
    }

    delete encoding_results_with_errors;
     
    delete decoding_results;

    delete[] g_tbls;
    delete[] encode_matrix;
}

TEST(Raid6, Raid6_testOneDeviceRebuildwithOneDataDeviceandOneParityDeviceErrorsCase01)
{
    // Given
    const PartitionPhysicalSize physicalSize{
        .startLba = 0 /* not interesting */,
        .lastLba = 0 /* not interesting */,
        .blksPerChunk = 64,
        .chunksPerStripe = 12,
        .stripesPerSegment = 0 /* not interesting */,
        .totalSegments = 0 /* not interesting */
    };

    // When
    uint32_t chunkCnt = physicalSize.chunksPerStripe; //# of total buffers
    uint32_t parityCnt = 2;                            //# of parity buffers
    uint32_t dataCnt = chunkCnt - parityCnt;                        //# of data buffers
    uint32_t chunkSize = physicalSize.blksPerChunk;
    
    vector<uint32_t> err_list_for_test;
    uint32_t firstErrIdx = 3;
    uint32_t secondErrIdx = chunkCnt -1;

    err_list_for_test.push_back(firstErrIdx);
    err_list_for_test.push_back(secondErrIdx);
    uint32_t nerrs = err_list_for_test.size();

    BYTE* encoding_sources[chunkCnt];
    BYTE* encoding_results_with_errors;
    BYTE* decoding_results;

    BYTE* encode_matrix = new BYTE[chunkCnt * dataCnt];
    BYTE* g_tbls = new BYTE[dataCnt * parityCnt * 32];

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        encoding_sources[i] = new BYTE[chunkSize];
    }

    encoding_results_with_errors = new BYTE[chunkSize * (chunkCnt - nerrs)];

    decoding_results = new BYTE[chunkSize * nerrs];

    unsigned int seed = time(NULL);
    for(uint32_t i = 0; i <dataCnt; i++)
    {
         for(uint32_t j = 0; j < chunkSize; j++)
        {
            encoding_sources[i][j] = rand_r(&seed);
        }
    }

    gf_gen_cauchy1_matrix(encode_matrix, chunkCnt, dataCnt);
    ec_init_tables(dataCnt, parityCnt, &encode_matrix[dataCnt * dataCnt], g_tbls);
    ec_encode_data(chunkSize, dataCnt, parityCnt, g_tbls, encoding_sources, &encoding_sources[dataCnt]);

    vector<ArrayDeviceState> vec;
    for(uint32_t i = 0; i < chunkCnt; i++)
    {
        if(i == firstErrIdx || i == secondErrIdx )
        {
            vec.push_back(ArrayDeviceState::FAULT);
        }
        else
        {
            vec.push_back(ArrayDeviceState::NORMAL);
        }

    }

    Raid6 raid6(&physicalSize, 0, [&vec](){
        return vec;
    });
    uint32_t tempInx = 0;
    vector<uint32_t> idx_for_test;
    for(uint32_t i = 0; i<chunkCnt; i++)
    {
        if(find(err_list_for_test.begin(), err_list_for_test.end(), i) == err_list_for_test.end())
        {
            idx_for_test.push_back(tempInx);
        }
        tempInx++;
    }

    uint32_t src_i = 0;
    for(auto idx:idx_for_test)
    {
        for(uint32_t j =0; j < chunkSize; j++)
        {
            encoding_results_with_errors[src_i*chunkSize + j] = encoding_sources[idx][j];
        }
        src_i++;
    }

    err_list_for_test.clear();
    err_list_for_test.push_back(firstErrIdx);
    raid6._RebuildData(decoding_results, encoding_results_with_errors, chunkSize, err_list_for_test,0);

    for(uint32_t i = 0; i < err_list_for_test.size(); i++)
    {
        ASSERT_EQ(0, memcmp(decoding_results, encoding_sources[err_list_for_test[i]], chunkSize));
    }

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        delete[] encoding_sources[i];
    }

    delete encoding_results_with_errors;
     
    delete decoding_results;

    delete[] g_tbls;
    delete[] encode_matrix;
}


TEST(Raid6, Raid6_testOneDeviceRebuildwithOneDataDeviceandOneParityDeviceErrorsCase02)
{
    // Given
    const PartitionPhysicalSize physicalSize{
        .startLba = 0 /* not interesting */,
        .lastLba = 0 /* not interesting */,
        .blksPerChunk = 64,
        .chunksPerStripe = 12,
        .stripesPerSegment = 0 /* not interesting */,
        .totalSegments = 0 /* not interesting */
    };

    // When
    uint32_t chunkCnt = physicalSize.chunksPerStripe; //# of total buffers
    uint32_t parityCnt = 2;                            //# of parity buffers
    uint32_t dataCnt = chunkCnt - parityCnt;                        //# of data buffers
    uint32_t chunkSize = physicalSize.blksPerChunk;
    
    vector<uint32_t> err_list_for_test;
    uint32_t firstErrIdx = 3;
    uint32_t secondErrIdx = chunkCnt -1;

    err_list_for_test.push_back(firstErrIdx);
    err_list_for_test.push_back(secondErrIdx);
    uint32_t nerrs = err_list_for_test.size();

    BYTE* encoding_sources[chunkCnt];
    BYTE* encoding_results_with_errors;
    BYTE* decoding_results;

    BYTE* encode_matrix = new BYTE[chunkCnt * dataCnt];
    BYTE* g_tbls = new BYTE[dataCnt * parityCnt * 32];

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        encoding_sources[i] = new BYTE[chunkSize];
    }

    encoding_results_with_errors = new BYTE[chunkSize * (chunkCnt - nerrs)];

    decoding_results = new BYTE[chunkSize * nerrs];

    unsigned int seed = time(NULL);
    for(uint32_t i = 0; i <dataCnt; i++)
    {
         for(uint32_t j = 0; j < chunkSize; j++)
        {
            encoding_sources[i][j] = rand_r(&seed);
        }
    }

    gf_gen_cauchy1_matrix(encode_matrix, chunkCnt, dataCnt);
    ec_init_tables(dataCnt, parityCnt, &encode_matrix[dataCnt * dataCnt], g_tbls);
    ec_encode_data(chunkSize, dataCnt, parityCnt, g_tbls, encoding_sources, &encoding_sources[dataCnt]);

    vector<ArrayDeviceState> vec;
    for(uint32_t i = 0; i < chunkCnt; i++)
    {
        if(i == firstErrIdx || i == secondErrIdx )
        {
            vec.push_back(ArrayDeviceState::FAULT);
        }
        else
        {
            vec.push_back(ArrayDeviceState::NORMAL);
        }

    }

    Raid6 raid6(&physicalSize, 0, [&vec](){
        return vec;
    });
    uint32_t tempInx = 0;
    vector<uint32_t> idx_for_test;
    for(uint32_t i = 0; i<chunkCnt; i++)
    {
        if(find(err_list_for_test.begin(), err_list_for_test.end(), i) == err_list_for_test.end())
        {
            idx_for_test.push_back(tempInx);
        }
        tempInx++;
    }

    uint32_t src_i = 0;
    for(auto idx:idx_for_test)
    {
        for(uint32_t j =0; j < chunkSize; j++)
        {
            encoding_results_with_errors[src_i*chunkSize + j] = encoding_sources[idx][j];
        }
        src_i++;
    }

    err_list_for_test.clear();
    err_list_for_test.push_back(secondErrIdx);
    raid6._RebuildData(decoding_results, encoding_results_with_errors, chunkSize, err_list_for_test,0);

    for(uint32_t i = 0; i < err_list_for_test.size(); i++)
    {
        ASSERT_EQ(0, memcmp(decoding_results, encoding_sources[err_list_for_test[i]], chunkSize));
    }

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        delete[] encoding_sources[i];
    }

    delete encoding_results_with_errors;
     
    delete decoding_results;

    delete[] g_tbls;
    delete[] encode_matrix;
}

TEST(Raid6, Raid6_testOneDeviceRebuildwithOneDataDeviceandOneParityDeviceErrorsafterMakeParity01)
{
    uint32_t STRIPEID = 3;
    uint32_t OFFSET = 0;
    // Given
    const PartitionPhysicalSize physicalSize{
        .startLba = 0 /* not interesting */,
        .lastLba = 0 /* not interesting */,
        .blksPerChunk = 4,
        .chunksPerStripe = 4,
        .stripesPerSegment = 0 /* not interesting */,
        .totalSegments = 0 /* not interesting */
    };
    
    // When
    uint32_t chunkCnt = physicalSize.chunksPerStripe; 
    uint32_t parityCnt = 2;
    uint32_t dataCnt = chunkCnt - parityCnt;
    uint32_t chunkSize = physicalSize.blksPerChunk;
    
    vector<uint32_t> err_list_for_test;
    uint32_t firstErrIdx = 0;
    uint32_t secondErrIdx = (STRIPEID + chunkCnt -2) % chunkCnt; //P Parity

    err_list_for_test.push_back(firstErrIdx);
    //err_list_for_test.push_back(secondErrIdx);
    uint32_t nerrs = err_list_for_test.size();
    /////
    BYTE* encoding_sources[chunkCnt];
    // BYTE* temp[chunkCnt];
    BYTE* ref_encoding_sources[chunkCnt];
    BYTE* encoding_results_with_errors;
    BYTE* decoding_results;

    BYTE* encode_matrix = new BYTE[chunkCnt * dataCnt];
    BYTE* g_tbls = new BYTE[dataCnt * parityCnt * 32];

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        encoding_sources[i] = new BYTE[chunkSize];
        ref_encoding_sources[i] = new BYTE[chunkSize];
        // temp[i] = new BYTE[chunkSize];
    }

    encoding_results_with_errors = new BYTE[chunkSize * (chunkCnt - nerrs)];
    decoding_results = new BYTE[chunkSize * nerrs];

    vector<ArrayDeviceState> vec;
    for(uint32_t i = 0; i < chunkCnt; i++)
    {
        if(i == firstErrIdx)
        {
            vec.push_back(ArrayDeviceState::FAULT);
        }
        else
        {
            vec.push_back(ArrayDeviceState::NORMAL);
        }
    }

    Raid6 raid6(&physicalSize, 0, [&vec](){
        return vec;
    });

    list<BufferEntry> buffers;
    int NUM_BLOCKS = physicalSize.blksPerChunk;

    BufferEntry be1 = generateRandomBufferEntry(NUM_BLOCKS, false);
    BufferEntry be2 = generateRandomBufferEntry(NUM_BLOCKS, false);

    buffers.push_back(be1);
    buffers.push_back(be2);

    uint32_t iter=0;
    BYTE* src_ptr = nullptr;
    for (const BufferEntry& src_buffer : buffers)
    {
        src_ptr = (BYTE*)src_buffer.GetBufferPtr();
        memcpy(encoding_sources[iter++], src_ptr, chunkSize);
    }

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        memcpy(ref_encoding_sources[i], encoding_sources[i], chunkSize);
    }

    LogicalBlkAddr lBlkAddr{
        .stripeId = STRIPEID,
        .offset = OFFSET
        };

    const LogicalWriteEntry& src{
        .addr = lBlkAddr,
        .blkCnt = physicalSize.blksPerChunk,
        .buffers = &buffers
        };

    list<BufferEntry> parities;

    BufferEntry pParity = generateInitializedBufferEntry(NUM_BLOCKS, true);
    BufferEntry qParity = generateInitializedBufferEntry(NUM_BLOCKS, true);

    parities.push_back(pParity);
    parities.push_back(qParity);
    
    raid6._ComputePQParities(parities, *(src.buffers));
    ASSERT_EQ(2, parities.size());

    memcpy(encoding_sources[dataCnt], (BYTE*)parities.front().GetBufferPtr(), chunkSize);
    memcpy(encoding_sources[dataCnt + 1], (BYTE*)parities.front().GetBufferPtr(), chunkSize);

    cout<<"1) encoding_results_with_parities using _ComputePQParities: "<<endl;
    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        for(uint32_t j = 0; j < chunkSize; j++)
        {
            printf("[%c]", encoding_sources[i][j]);
        }
        cout <<endl;
    }
    // //for reference
    // gf_gen_cauchy1_matrix(encode_matrix, chunkCnt, dataCnt);
    // ec_init_tables(dataCnt, parityCnt, &encode_matrix[dataCnt * dataCnt], g_tbls);
    // ec_encode_data(chunkSize, dataCnt, parityCnt, g_tbls, ref_encoding_sources, &ref_encoding_sources[dataCnt]);

    // cout<<"2) [ref] encoding_results_with_parities using ISAL: "<<endl;
    // for (uint32_t i = 0; i < chunkCnt; i++)
    // {
    //     for(uint32_t j = 0; j < chunkSize; j++)
    //     {
    //         printf("[%c]", ref_encoding_sources[i][j]);
    //     }
    //     cout <<endl;
    // }

    //  for (uint32_t i = 0; i < chunkCnt; i++)
    // {
    //     ASSERT_EQ(0, memcmp(encoding_sources[i],ref_encoding_sources[i],chunkSize));
    // }

    uint32_t tempInx = 0;
    vector<uint32_t> idx_for_test;
    for(uint32_t i = 0; i<chunkCnt; i++)
    {
        if(find(err_list_for_test.begin(), err_list_for_test.end(), i) == err_list_for_test.end())
        {
            idx_for_test.push_back(tempInx);
        }
        tempInx++;
    }

    uint32_t src_i = 0;
    for(auto idx:idx_for_test)
    {
        for(uint32_t j =0; j < chunkSize; j++)
        {
            encoding_results_with_errors[src_i*chunkSize + j] = encoding_sources[idx][j];
        }
        src_i++;
    }

    cout<<"2) encoding_results_with_errors"<<endl;
    for (uint32_t i = 0; i < (chunkCnt-nerrs)*chunkSize; i++)
    {
        printf("[%c]",encoding_results_with_errors[i]);
        if (i == chunkSize -1)
        {
            cout <<endl;
        }
    }

    // memcpy(temp[0], encoding_results_with_errors, chunkSize);
    // memcpy(encoding_results_with_errors, encoding_results_with_errors+chunkSize, chunkSize);
    // memcpy(encoding_results_with_errors+chunkSize, encoding_results_with_errors+chunkSize*2, chunkSize);
    // memcpy(encoding_results_with_errors+chunkSize*2, temp[0], chunkSize);

    // cout<<"4) encoding_results_with_errors_shuffle"<<endl;
    // for (uint32_t i = 0; i < (chunkCnt-nerrs)*chunkSize; i++)
    // {
    //     printf("[%c]",encoding_results_with_errors[i]);
    //     if (i == chunkSize -1)
    //     {
    //         cout <<endl;
    //     }
    // }

    cout <<endl;
    err_list_for_test.clear();
    err_list_for_test.push_back(firstErrIdx);
    raid6._RebuildData(decoding_results, encoding_results_with_errors, chunkSize, err_list_for_test, STRIPEID);
    cout << "****Stripe ID :"<< STRIPEID << "Decoding Target : " <<err_list_for_test[0]<<endl;
    cout<<"3) decoding_results after _RebuildData"<<endl;
    for (uint32_t i = 0; i < err_list_for_test.size()*chunkSize; i++)
    {
        printf("[%c]", decoding_results[i]);
          if ((i % chunkCnt) == (chunkCnt -1))
          {
                cout <<endl;
          }
    }
    cout <<endl;

    for(uint32_t i = 0; i < err_list_for_test.size(); i++)
    {
        ASSERT_EQ(0, memcmp(decoding_results, encoding_sources[err_list_for_test[i]], chunkSize));
    }

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        delete[] encoding_sources[i];
        delete[] ref_encoding_sources[i];
    }

    delete encoding_results_with_errors;
    delete decoding_results;

    delete[] g_tbls;
    delete[] encode_matrix;
}

TEST(Raid6, Raid6_testOneDeviceRebuildwithOneDataDeviceandOneParityDeviceErrorsafterMakeParity02)
{
    uint32_t STRIPEID = 3;
    uint32_t OFFSET = 0;
    // Given
    const PartitionPhysicalSize physicalSize{
        .startLba = 0 /* not interesting */,
        .lastLba = 0 /* not interesting */,
        .blksPerChunk = 4,
        .chunksPerStripe = 4,
        .stripesPerSegment = 0 /* not interesting */,
        .totalSegments = 0 /* not interesting */
    };
    
    // When
    uint32_t chunkCnt = physicalSize.chunksPerStripe; 
    uint32_t parityCnt = 2;
    uint32_t dataCnt = chunkCnt - parityCnt;
    uint32_t chunkSize = physicalSize.blksPerChunk;
    
    vector<uint32_t> err_list_for_test;
    uint32_t firstErrIdx = 0;
    uint32_t secondErrIdx = (STRIPEID + chunkCnt -2) % chunkCnt; //P Parity

    err_list_for_test.push_back(firstErrIdx);
    err_list_for_test.push_back(secondErrIdx);
    uint32_t nerrs = err_list_for_test.size();
    /////
    BYTE* encoding_sources[chunkCnt];
    BYTE* ref_encoding_sources[chunkCnt];
    BYTE* encoding_results_with_errors;
    BYTE* decoding_results;

    BYTE* encode_matrix = new BYTE[chunkCnt * dataCnt];
    BYTE* g_tbls = new BYTE[dataCnt * parityCnt * 32];

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        encoding_sources[i] = new BYTE[chunkSize];
        ref_encoding_sources[i] = new BYTE[chunkSize];
    }

    encoding_results_with_errors = new BYTE[chunkSize * (chunkCnt - nerrs)];
    decoding_results = new BYTE[chunkSize * nerrs];

    vector<ArrayDeviceState> vec;
    for(uint32_t i = 0; i < chunkCnt; i++)
    {
        if(i == firstErrIdx || i == secondErrIdx )
        {
            vec.push_back(ArrayDeviceState::FAULT);
        }
        else
        {
            vec.push_back(ArrayDeviceState::NORMAL);
        }
    }

    Raid6 raid6(&physicalSize, 0, [&vec](){
        return vec;
    });

    list<BufferEntry> buffers;
    int NUM_BLOCKS = physicalSize.blksPerChunk;

    BufferEntry be1 = generateRandomBufferEntry(NUM_BLOCKS, false);
    BufferEntry be2 = generateRandomBufferEntry(NUM_BLOCKS, false);

    buffers.push_back(be1);
    buffers.push_back(be2);

    uint32_t iter=0;
    BYTE* src_ptr = nullptr;
    for (const BufferEntry& src_buffer : buffers)
    {
        src_ptr = (BYTE*)src_buffer.GetBufferPtr();
        memcpy(encoding_sources[iter++], src_ptr, chunkSize);
    }

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        memcpy(ref_encoding_sources[i], encoding_sources[i], chunkSize);
    }

    LogicalBlkAddr lBlkAddr{
        .stripeId = STRIPEID,
        .offset = OFFSET
        };

    const LogicalWriteEntry& src{
        .addr = lBlkAddr,
        .blkCnt = physicalSize.blksPerChunk,
        .buffers = &buffers
        };

    list<BufferEntry> parities;

    BufferEntry pParity = generateInitializedBufferEntry(NUM_BLOCKS, true);
    BufferEntry qParity = generateInitializedBufferEntry(NUM_BLOCKS, true);

    parities.push_back(pParity);
    parities.push_back(qParity);
    
    raid6._ComputePQParities(parities, *(src.buffers));
    ASSERT_EQ(2, parities.size());

    memcpy(encoding_sources[dataCnt], (BYTE*)parities.front().GetBufferPtr(), chunkSize);
    memcpy(encoding_sources[dataCnt + 1], (BYTE*)parities.front().GetBufferPtr(), chunkSize);

    cout<<"1) encoding_results_with_parities using _ComputePQParities: "<<endl;
    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        for(uint32_t j = 0; j < chunkSize; j++)
        {
            printf("[%c]", encoding_sources[i][j]);
        }
        cout <<endl;
    }
    //for reference
    // gf_gen_cauchy1_matrix(encode_matrix, chunkCnt, dataCnt);
    // ec_init_tables(dataCnt, parityCnt, &encode_matrix[dataCnt * dataCnt], g_tbls);
    // ec_encode_data(chunkSize, dataCnt, parityCnt, g_tbls, ref_encoding_sources, &ref_encoding_sources[dataCnt]);

    // cout<<"2) [ref] encoding_results_with_parities using ISAL: "<<endl;
    // for (uint32_t i = 0; i < chunkCnt; i++)
    // {
    //     for(uint32_t j = 0; j < chunkSize; j++)
    //     {
    //         printf("[%c]", ref_encoding_sources[i][j]);
    //     }
    //     cout <<endl;
    // }

    //  for (uint32_t i = 0; i < chunkCnt; i++)
    // {
    //     ASSERT_EQ(0, memcmp(encoding_sources[i],ref_encoding_sources[i],chunkSize));
    // }

    uint32_t tempInx = 0;
    vector<uint32_t> idx_for_test;
    for(uint32_t i = 0; i<chunkCnt; i++)
    {
        if(find(err_list_for_test.begin(), err_list_for_test.end(), i) == err_list_for_test.end())
        {
            idx_for_test.push_back(tempInx);
        }
        tempInx++;
    }

    uint32_t src_i = 0;
    for(auto idx:idx_for_test)
    {
        for(uint32_t j =0; j < chunkSize; j++)
        {
            encoding_results_with_errors[src_i*chunkSize + j] = encoding_sources[idx][j];
        }
        src_i++;
    }

    cout<<"2) encoding_results_with_errors"<<endl;
    for (uint32_t i = 0; i < (chunkCnt-nerrs)*chunkSize; i++)
    {
        printf("[%c]",encoding_results_with_errors[i]);
          if ((i % chunkCnt) == (chunkCnt -1))
          {
                cout <<endl;
          }
    }
    cout <<endl;
    err_list_for_test.clear();
    err_list_for_test.push_back(firstErrIdx);
    raid6._RebuildData(decoding_results, encoding_results_with_errors, chunkSize, err_list_for_test, STRIPEID);

    cout << "****Stripe ID :"<< STRIPEID << "Decoding Target : " <<err_list_for_test[0]<<endl;
    cout<<"3) decoding_results after _RebuildData"<<endl;
    for (uint32_t i = 0; i < err_list_for_test.size()*chunkSize; i++)
    {
        printf("[%c]", decoding_results[i]);
        if (i == chunkCnt -1)
            cout <<endl;
    }

    for(uint32_t i = 0; i < err_list_for_test.size(); i++)
    {
        ASSERT_EQ(0, memcmp(decoding_results, encoding_sources[err_list_for_test[i]], chunkSize));
    }

    for (uint32_t i = 0; i < chunkCnt; i++)
    {
        delete[] encoding_sources[i];
        delete[] ref_encoding_sources[i];
    }

    delete encoding_results_with_errors;
    delete decoding_results;

    delete[] g_tbls;
    delete[] encode_matrix;
}

} // namespace pos
