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

    std::unique_ptr<NDArray<float>> arrayExp(new NDArray<float>('c', {2, 1, 4, 4}));
    std::unique_ptr<NDArray<float>> arrayBad(new NDArray<float>('c', {2, 1, 4, 4}));

    arrayExp->setBuffer(arraySource->getBuffer() + 50);
    arrayExp->printShapeInfo("Exp shapeBuffer: ");


    std::vector<int> badShape({4, 2, 1, 4, 4, 80, 16, 4, 1, 0, -1, 99});

    arrayBad->setBuffer(arraySource->getBuffer() + 50);
    arrayBad->setShapeInfo(badShape.data());
    arrayBad->printShapeInfo("Bad shapeBuffer: ");

    std::unique_ptr<NDArray<float>> sumExp(arrayExp->sum({1}));
    std::unique_ptr<NDArray<float>> sumBad(arrayBad->sum({1}));


    sumExp->printBuffer("Proper result:");
    sumBad->printBuffer("Wrong  result:");

    ASSERT_TRUE(sumBad->equalsTo(sumExp.get()));
}

#endif //LIBND4J_TADTESTS_H
