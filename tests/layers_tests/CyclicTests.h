//
// Created by raver119 on 19.09.17.
//

#ifndef LIBND4J_CYCLICTESTS_H
#define LIBND4J_CYCLICTESTS_H

#include "testlayers.h"
#include <NDArray.h>
#include <Block.h>
#include <Node.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <ops/declarable/declarable_ops.h>
#include <ops/declarable/generic/convo/convo_ops.h>

using namespace nd4j::graph;

class CyclicTests : public testing::Test {
public:
    int numLoops = 1000000;

    int extLoops = 1000;
    int intLoops = 1000;
};

// simple test for NDArray leaks
TEST_F(CyclicTests, TestNDArray1) {

    for (int i = 0; i < numLoops; i++) {
        auto arr = new NDArray<float>(1000, 1000, 'c');
        arr->assign(1.0);
        delete arr;
    }
}

TEST_F(CyclicTests, TestNDArrayWorkspace1) {

    for (int e = 0; e < extLoops; e++) {
        nd4j::memory::Workspace workspace(10000000);
        nd4j::memory::MemoryRegistrator::getInstance()->attachWorkspace(&workspace);

        for (int i = 0; i < intLoops; i++) {
            workspace.scopeIn();
            
            auto arr = new NDArray<float>(1000, 1000, 'c');
            
            delete arr;
            
            workspace.scopeOut();
        }
    }
}


#endif //LIBND4J_CYCLICTESTS_H
