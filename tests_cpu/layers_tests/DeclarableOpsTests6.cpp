//
// Created by raver119 on 09.02.18.
//


#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>


using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests6 : public testing::Test {
public:

    DeclarableOpsTests6() {
        printf("\n");
        fflush(stdout);
    }
};


TEST_F(DeclarableOpsTests6, Test_Dilation2D_Again_1) {
    NDArray<float> x('c', {4, 128, 128, 4});
    NDArray<float> w('c', {4, 5, 4});

    nd4j::ops::dilation2d<float> op;
    auto result = op.execute({&x, &w}, {}, {1, 1,5,7,1, 1,2,3,1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    delete result;
}


TEST_F(DeclarableOpsTests6, Test_Dilation2D_Again_2) {
    NDArray<float> x('c', {4, 26, 19, 4});
    NDArray<float> w('c', {11, 7, 4});

    nd4j::ops::dilation2d<float> op;
    auto result = op.execute({&x, &w}, {}, {0, 1,2,3,1, 1,3,2,1});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}
