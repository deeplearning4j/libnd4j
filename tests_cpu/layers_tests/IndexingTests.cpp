//
// Created by raver119 on 31.10.2017.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <NativeOps.h>

using namespace nd4j;
using namespace nd4j::graph;

class IndexingTests : public testing::Test {
public:

};

TEST_F(IndexingTests, StridedSlice_1) {
    NDArray<float> x('c', {3, 3, 3});
    NDArray<float> exp('c', {1, 1, 3});
    exp.putScalar(0, 25.f);
    exp.putScalar(1, 26.f);
    exp.putScalar(2, 27.f);

    NDArrayFactory<float>::linspace(1, x);

    //nd4j_debug("print x->rankOf(): %i", x.rankOf());

    /*
    auto tads = NDArrayFactory<float>::allTensorsAlongDimension(&x, {0});
    nd4j_debug("numTads: %i\n", tads->size());
    for (int e = 0; e < tads->size(); e++)
        tads->at(e)->assign((float) e);
    */

    nd4j::ops::strided_slice<float> op;

    auto result = op.execute({&x}, {}, {2,2,0,  3,3,3,  1,1,1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printShapeInfo("z shape");
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));
}


TEST_F(IndexingTests, StridedSlice_2) {
    NDArray<float> x('c', {5, 5, 5});

    float _expB[] = {86.f,   87.f,   88.f,  91.f,   92.f,   93.f,  96.f,   97.f,   98.f,   111.f,  112.f,  113.f,  116.f,  117.f,  118.f,  121.f,  122.f,  123.f,};
    NDArray<float> exp('c', {2, 3, 3});
    exp.setBuffer(_expB);

    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::strided_slice<float> op;

    auto result = op.execute({&x}, {}, {3,2,0,  5,5,3,  1,1,1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printShapeInfo("z shape");
    //z->printBuffer("z buffer");
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));
}


TEST_F(IndexingTests, StridedSlice_3) {
    NDArray<float> x('c', {5, 5, 5});

    float _expB[] = {86.f, 88.f,  91.f, 93.f, 96.f, 98.f, 111.f,  113.f,  116.f, 118.f,  121.f,  123.f,};
    NDArray<float> exp('c', {2, 3, 2});
    exp.setBuffer(_expB);

    NDArrayFactory<float>::linspace(1, x);

    nd4j::ops::strided_slice<float> op;

    auto result = op.execute({&x}, {}, {3,2,0,  5,5,3,  1,1,2});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    z->printShapeInfo("z shape");
    z->printBuffer("z buffer");
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));
}