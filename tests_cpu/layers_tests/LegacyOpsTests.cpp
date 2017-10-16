//
// Created by raver119 on 16.10.2017.
//

#include "testlayers.h"
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <ops/declarable/LegacyTransformOp.h>
#include <ops/declarable/LegacyPairwiseTransformOp.h>
#include <ops/declarable/LegacyScalarOp.h>

using namespace nd4j;
using namespace nd4j::ops;

class LegacyOpsTests : public testing::Test {

};


TEST_F(LegacyOpsTests, TransformTests_1) {
    NDArray<float> x('c', {5, 5});
    x.assign(1.0);

    NDArray<float> exp('c', {5, 5});
    exp.assign(-1.0);

    nd4j::ops::LegacyTransformOp<float> op(6);
    op.execute({&x}, {&x}, {}, {});

    ASSERT_TRUE(x.equalsTo(&exp));
}

TEST_F(LegacyOpsTests, TransformTests_2) {
    NDArray<float> x('c', {5, 5});
    x.assign(1.0);

    NDArray<float> exp('c', {5, 5});
    exp.assign(-1.0);

    nd4j::ops::LegacyTransformOp<float> op(6);
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(1, result->size());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(LegacyOpsTests,  PWT_Tests_1) {
    NDArray<float> x('c', {5, 5});
    x.assign(2.0);

    NDArray<float> y('c', {5, 5});
    y.assign(3.0);

    NDArray<float> exp('c', {5, 5});
    exp.assign(6.0);

    nd4j::ops::LegacyPairwiseTransformOp<float> op(6);
    Nd4jStatus status = op.execute({&x, &y}, {&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(exp.equalsTo(&x));


}

TEST_F(LegacyOpsTests,  PWT_Tests_2) {
    NDArray<float> x('c', {5, 5});
    x.assign(2.0);

    NDArray<float> y('c', {5, 5});
    y.assign(3.0);

    NDArray<float> exp('c', {5, 5});
    exp.assign(6.0);

    nd4j::ops::LegacyPairwiseTransformOp<float> op(6);
    auto result = op.execute({&x, &y}, {}, {});

    auto z = result->at(0);

    z->printBuffer("Z");
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(LegacyOpsTests, Scalar_Test_1) {
    NDArray<float> x('c', {5, 5});
    x.assign(2.0);

    NDArray<float> exp('c', {5, 5});
    exp.assign(7.0);

    nd4j::ops::LegacyScalarOp<float> op(0);
    op.execute({&x}, {&x}, {5.0}, {});

    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(LegacyOpsTests, Scalar_Test_2) {
    NDArray<float> x('c', {5, 5});
    x.assign(2.0);

    NDArray<float> exp('c', {5, 5});
    exp.assign(7.0);

    nd4j::ops::LegacyScalarOp<float> op(0, 5.0f);
    auto result = op.execute({&x}, {}, {});

    auto z = result->at(0);
    ASSERT_TRUE(exp.equalsTo(z));
}