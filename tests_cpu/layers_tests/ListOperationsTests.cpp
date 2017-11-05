//
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::ops;

class ListOperationsTests : public testing::Test {

};

TEST_F(ListOperationsTests, BasicTest_1) {
    NDArrayList<double> list;
    NDArray<double> x('c', {1, 128});
    NDArrayFactory<double>::linspace(1, x);

    nd4j::ops::write_list<double> op;

    op.execute(&list, {&x}, {}, {1});

    ASSERT_EQ(1, list.elements());

    op.execute(&list, {&x}, {}, {2});

    ASSERT_EQ(2, list.elements());
}
