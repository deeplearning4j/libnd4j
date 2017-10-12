//
// Created by raver119 on 12.10.2017.
//

#include "testlayers.h"
#include <NDArray.h>
#include <ops/declarable/CustomOperations.h>


using namespace nd4j;
using namespace nd4j::ops;

class ParityOpsTests : public testing::Test {
public:

};


TEST_F(ParityOpsTests, TestZeroAs1) {
    NDArray<float> x('c', {10, 10});
    x.assign(1.0);

    NDArray<float> exp('c', {10, 10});
    exp.assign(0.0f);

    nd4j::ops::zeros_as<float> op;

    auto result = op.execute({&x}, {}, {});

    auto z = result->at(0);

    ASSERT_TRUE(z->isSameShape(&x));
    ASSERT_TRUE(z->equalsTo(&exp));

    delete result;
}

TEST_F(ParityOpsTests, TestMaximum1) {
    NDArray<float> x('c', {10, 10});
    x.assign(1.0);

    NDArray<float> y('c', {10, 10});
    y.assign(2.0);

    nd4j::ops::maximum<float> op;

    auto result = op.execute({&x, &y}, {}, {});

    auto z = result->at(0);

    ASSERT_TRUE(y.equalsTo(z));

    delete result;
}


TEST_F(ParityOpsTests, TestMinimum1) {
    NDArray<float> x('c', {10, 10});
    x.assign(1.0f);

    NDArray<float> y('c', {10, 10});
    y.assign(-2.0f);


    nd4j::ops::minimum<float> op;

    auto result = op.execute({&x, &y}, {}, {});

    auto z = result->at(0);

    ASSERT_TRUE(y.equalsTo(z));

    delete result;
}