//
//  @author raver119@gmail.com
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <NativeOps.h>
#include <helpers/BitwiseUtils.h>

using namespace nd4j;
using namespace nd4j::graph;

class ScalarTests : public testing::Test {
public:

};

TEST_F(ScalarTests, Test_Create_1) {
    NDArray<float> x(2.0f);

    ASSERT_EQ(0, x.rankOf());
    ASSERT_EQ(1, x.lengthOf());
    ASSERT_TRUE(x.isScalar());
    ASSERT_FALSE(x.isVector());
    ASSERT_FALSE(x.isRowVector());
    ASSERT_FALSE(x.isColumnVector());
    ASSERT_FALSE(x.isMatrix());
}

TEST_F(ScalarTests, Test_Add_1) {
    NDArray<float> x(2.0f);
    NDArray<float> exp(5.0f);

    x += 3.0f;

    ASSERT_NEAR(5.0f, x.getScalar(0), 1e-5f);
    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(ScalarTests, Test_EQ_1) {
    NDArray<float> x(2.0f);
    NDArray<float> y(3.0f);

    ASSERT_TRUE(y.isSameShape(&x));
    ASSERT_FALSE(y.equalsTo(&x));
}