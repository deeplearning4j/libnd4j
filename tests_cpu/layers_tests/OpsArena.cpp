//
// Created by raver119 on 11.10.2017.
//
// This "set of tests" is special one - we don't check ops results here. we just check for memory equality BEFORE op launch and AFTER op launch
//
//
#include "testlayers.h"
#include <vector>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/OpTuple.h>
#include <ops/declarable/OpRegistrator.h>
#include <memory/MemoryReport.h>
#include <memory/MemoryUtils.h>

using namespace nd4j;
using namespace nd4j::ops;

class OpsArena : public testing::Test {
public:
    const int numIterations = 100000;
    std::vector<OpTuple> tuples;


    OpsArena() {
        printf("\nStarting memory tests...\n");



        // mergemax
        auto mergeMax_X0 = new NDArray<float>('c', {100, 100});
        auto mergeMax_X1 = new NDArray<float>('c', {100, 100});
        auto mergeMax_X2 = new NDArray<float>('c', {100, 100});
        auto mergeMax_Tuple = new OpTuple("mergemax", {mergeMax_X0, mergeMax_X1, mergeMax_X2}, {}, {});
        tuples.push_back(*mergeMax_Tuple);

    }
};


TEST_F(OpsArena, TestFeedForward) {

    for (auto tuple: tuples) {
        auto op = OpRegistrator::getInstance()->getOperationFloat(tuple._opName);
        if (op == nullptr) {
            nd4j_printf("Can't find Op by name: [%s]\n", tuple._opName);
            ASSERT_TRUE(false);
        }

        nd4j_printf("Testing op [%s]", tuple._opName);
        nd4j::memory::MemoryReport before, after;

        auto tmp = op->execute(tuple._inputs, tuple._tArgs, tuple._iArgs);
        delete tmp;

        auto b = nd4j::memory::MemoryUtils::retrieveMemoryStatistics(before);
        if (!b)
            ASSERT_TRUE(false);

        for (int e = 0; e < numIterations; e++) {
            auto result = op->execute(tuple._inputs, tuple._tArgs, tuple._iArgs);

            // we just want to be sure op was executed successfully
            ASSERT_TRUE(result->size() > 0);

            delete result;
        }


        auto a = nd4j::memory::MemoryUtils::retrieveMemoryStatistics(after);
        if (!a)
            ASSERT_TRUE(false);


        // this is our main assertion. memory footprint after op run should NOT be higher then before
        if (after > before) {
            nd4j_printf("OpName: [%s]; RSS before: [%lld]; RSS after: [%lld]\n", tuple._opName, before.getRSS(), after.getRSS())
            ASSERT_TRUE(after <= before);
        }
    }
}

