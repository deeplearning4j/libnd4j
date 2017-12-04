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
    NDArray<float> x0('c', {10, 10});
    NDArray<float> x1('c', {10, 10});
    NDArray<float> nexp0('c', {10, 10});
    NDArray<float> nexp1('c', {10, 10});
    NDArray<float> nexp2('c', {10, 10});

    nexp0.assign(-1.0f);
    nexp1.assign(-2.0f);
    nexp2.assign(-3.0f);

    NDArrayFactory<float>::linspace(1, x0);
    NDArrayFactory<float>::linspace(1, x1);

    float prob[] = {0.5f};

    x0.template applyRandom<randomOps::DropOut<float>>(_rngA, nullptr, &x0, prob);
    x1.template applyRandom<randomOps::DropOut<float>>(_rngB, nullptr, &x1, prob);

    ASSERT_TRUE(x0.equalsTo(&x1));

    // this check is required to ensure we're calling wrong signature
    ASSERT_FALSE(x0.equalsTo(&nexp0));
    ASSERT_FALSE(x0.equalsTo(&nexp1));
    ASSERT_FALSE(x0.equalsTo(&nexp2));
}

TEST_F(RNGTests, Test_DropoutInverted_1) {
    NDArray<float> x0('c', {10, 10});
    NDArray<float> x1('c', {10, 10});
    NDArray<float> nexp0('c', {10, 10});
    NDArray<float> nexp1('c', {10, 10});
    NDArray<float> nexp2('c', {10, 10});

    nexp0.assign(-1.0f);
    nexp1.assign(-2.0f);
    nexp2.assign(-3.0f);

    NDArrayFactory<float>::linspace(1, x0);
    NDArrayFactory<float>::linspace(1, x1);

    float prob[] = {0.5f};

    x0.template applyRandom<randomOps::DropOutInverted<float>>(_rngA, nullptr, &x0, prob);
    x1.template applyRandom<randomOps::DropOutInverted<float>>(_rngB, nullptr, &x1, prob);

    ASSERT_TRUE(x0.equalsTo(&x1));

    // this check is required to ensure we're calling wrong signature
    ASSERT_FALSE(x0.equalsTo(&nexp0));
    ASSERT_FALSE(x0.equalsTo(&nexp1));
    ASSERT_FALSE(x0.equalsTo(&nexp2));
}