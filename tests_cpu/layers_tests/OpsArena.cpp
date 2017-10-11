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
    const int numIterations = 10;
    std::vector<OpTuple *> tuples;


    OpsArena() {
        nd4j_printf("\nStarting memory tests...\n","");


        // conv2d_bp
        auto conv2d_bp_Input = new NDArray<float>('c', {2, 1, 4, 4});
        auto conv2d_bp_Weights = new NDArray<float>('c', {2, 1, 3, 3});
        auto conv2d_bp_Bias = new NDArray<float>('c', {2, 1});
        auto conv2d_bp_Epsilon = new NDArray<float > ('c', {2, 2, 4, 4});
        tuples.push_back(new OpTuple("conv2d_bp", {conv2d_bp_Input, conv2d_bp_Weights, conv2d_bp_Bias, conv2d_bp_Epsilon}, {}, {3, 3, 1, 1, 0, 0, 1, 1, 1}));


        // mergeavg
        tuples.emplace_back((new OpTuple("mergeavg"))
                                    ->addInput(new NDArray<float>('c', {100, 100}))
                                    ->addInput(new NDArray<float>('c', {100, 100}))
                                    ->addInput(new NDArray<float>('c', {100, 100}))
                                    ->addInput(new NDArray<float >('c', {100, 100})));

        // mergemax
        auto mergeMax_X0 = new NDArray<float>('c', {100, 100});
        auto mergeMax_X1 = new NDArray<float>('c', {100, 100});
        auto mergeMax_X2 = new NDArray<float>('c', {100, 100});
        tuples.push_back(new OpTuple("mergemax", {mergeMax_X0, mergeMax_X1, mergeMax_X2}, {}, {}));

        // conv2d
        auto conv2d_Input = new NDArray<float>('c', {1, 2, 5, 4});
        auto conv2d_Weights = new NDArray<float>('c', {3, 2, 2, 2});
        auto conv2d_Bias = new NDArray<float>('c', {3, 1});
        tuples.push_back(new OpTuple("conv2d", {conv2d_Input, conv2d_Weights, conv2d_Bias}, {}, {2, 2, 1, 1, 0, 0, 1, 1, 1}));


        // test custom op
        tuples.emplace_back((new OpTuple("testcustom"))
                                    ->setIArgs({1, 2})
                                    ->addInput(new NDArray<float>('c', {100, 100})));


        // deconv2d
        tuples.emplace_back((new OpTuple("deconv2d"))
                                    ->addInput(new NDArray<float>('c', {2, 3, 4, 4}))
                                    ->addInput(new NDArray<float >('c', {3, 3, 5, 5}))
                                    ->setIArgs({5, 5, 1, 1, 0, 0, 1, 1, 0}));

        // maxpool2d
        tuples.emplace_back((new OpTuple("maxpool2d"))
                                    ->addInput(new NDArray<float>('c', {2, 1, 28, 28}))
                                    ->setIArgs({5, 5, 1, 1, 0, 0, 2, 2, 0}));
    }
};


TEST_F(OpsArena, TestFeedForward) {

    for (auto tuple: tuples) {
        auto op = OpRegistrator::getInstance()->getOperationFloat(tuple->_opName);
        if (op == nullptr) {
            nd4j_printf("Can't find Op by name: [%s]\n", tuple->_opName);
            ASSERT_TRUE(false);
        }

        nd4j_printf("Testing op [%s]\n", tuple->_opName);
        nd4j::memory::MemoryReport before, after;

        // warmup
        auto tmp1 = op->execute(tuple->_inputs, tuple->_tArgs, tuple->_iArgs);
        auto tmp2 = op->execute(tuple->_inputs, tuple->_tArgs, tuple->_iArgs);
        delete tmp1;
        delete tmp2;

        auto b = nd4j::memory::MemoryUtils::retrieveMemoryStatistics(before);

        if (!b)
            ASSERT_TRUE(false);

        for (int e = 0; e < numIterations; e++) {
            auto result = op->execute(tuple->_inputs, tuple->_tArgs, tuple->_iArgs);

            // we just want to be sure op was executed successfully
            ASSERT_TRUE(result->size() > 0);

            delete result;
        }


        auto a = nd4j::memory::MemoryUtils::retrieveMemoryStatistics(after);
        if (!a)
            ASSERT_TRUE(false);


        // this is our main assertion. memory footprint after op run should NOT be higher then before
        if (after > before) {
            nd4j_printf("WARNING!!! OpName: [%s]; RSS before: [%lld]; RSS after: [%lld]\n", tuple->_opName, before.getRSS(), after.getRSS())
        //    ASSERT_TRUE(after <= before);
        }
    }
}



TEST_F(OpsArena, TestMmulHelper1) {
    NDArray<float> a('c', {100, 100});
    NDArray<float> b('c', {100, 100});
    NDArray<float> c('c', {100, 100});

    nd4j::NDArrayFactory<float>::mmulHelper(&a, &b, &c);

    nd4j::memory::MemoryReport before, after;

    nd4j::memory::MemoryUtils::retrieveMemoryStatistics(before);

    for (int e = 0; e < numIterations; e++) {
        nd4j::NDArrayFactory<float>::mmulHelper(&a, &b, &c);
    }

    nd4j::memory::MemoryUtils::retrieveMemoryStatistics(after);
    if (after > before) {
        nd4j_printf("WARNING!!! OpName: [%s]; RSS before: [%lld]; RSS after: [%lld]\n", "mmulHelper", before.getRSS(), after.getRSS())
        ASSERT_TRUE(after <= before);
    }
}


TEST_F(OpsArena, TestMmulHelper2) {
    NDArray<float> a('c', {100, 100});
    NDArray<float> b('c', {100, 100});

    auto c = nd4j::NDArrayFactory<float>::mmulHelper(&a, &b);
    delete c;

    nd4j::memory::MemoryReport before, after;

    nd4j::memory::MemoryUtils::retrieveMemoryStatistics(before);

    for (int e = 0; e < numIterations; e++) {
        c = nd4j::NDArrayFactory<float>::mmulHelper(&a, &b);
        delete c;
    }

    nd4j::memory::MemoryUtils::retrieveMemoryStatistics(after);
    if (after > before) {
        nd4j_printf("WARNING!!! OpName: [%s]; RSS before: [%lld]; RSS after: [%lld]\n", "mmulHelper", before.getRSS(), after.getRSS())
        ASSERT_TRUE(after <= before);
    }
}

