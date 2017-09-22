//
// @author raver119@gmail.com
//

#ifndef LIBND4J_TADTESTS_H
#define LIBND4J_TADTESTS_H

#include "testlayers.h"
#include <NDArray.h>

class TadTests : public testing::Test {
public:
    int numLoops = 100000000;

    int extLoops = 1000;
    int intLoops = 1000;
};

TEST_F(TadTests, Test4DTad1) {
    std::unique_ptr<NDArray<float>> arraySource(NDArrayFactory::linspace(1.0f, 10000.0f, 10000));


}

#endif //LIBND4J_TADTESTS_H
