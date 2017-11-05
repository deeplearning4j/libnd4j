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

    nd4j::ops::write_list<double> op;
}
