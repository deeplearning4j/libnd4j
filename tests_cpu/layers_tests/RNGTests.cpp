//
//  @author raver119@gmail.com
//

#include "testlayers.h"
#include <chrono>
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;

class RNGTests : public testing::Test {
private:
    NativeOps nativeOps;
    Nd4jIndex *_bufferA;
    Nd4jIndex *_bufferB;

public:
    long _seed = 119L;
    nd4j::random::RandomBuffer *_rngA;
    nd4j::random::RandomBuffer *_rngB;

    RNGTests() {
        _bufferA = new Nd4jIndex[100000];
        _bufferB = new Nd4jIndex[100000];
        _rngA = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, _seed, 100000, (Nd4jPointer) _bufferA);
        _rngB = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, _seed, 100000, (Nd4jPointer) _bufferB);
    }

    ~RNGTests() {
        nativeOps.destroyRandom(_rngA);
        nativeOps.destroyRandom(_rngB);
        delete[] _bufferA;
        delete[] _bufferB;
    }
};

TEST_F(RNGTests, Test_Dropout_1) {
    NDArray<float> x('c', {10, 10});

    NDArrayFactory<float>::linspace(1, x);
}