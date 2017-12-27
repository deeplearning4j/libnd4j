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

class SingleDimTests : public testing::Test {
public:

};

TEST_F(SingleDimTests, Test_Create_1) {
    NDArray<float> x('c', {5}, {1, 2, 3, 4, 5});
    ASSERT_EQ(5, x.lengthOf());
    ASSERT_EQ(1, x.rankOf());
    ASSERT_TRUE(x.isVector());
    ASSERT_TRUE(x.isRowVector());
}