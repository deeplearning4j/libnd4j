//
// Created by raver119 on 09.02.18.
//


#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>


using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests6 : public testing::Test {
public:

    DeclarableOpsTests6() {
        printf("\n");
        fflush(stdout);
    }
};


TEST_F(DeclarableOpsTests6, Test_Dilation2D_Again_1) {
    NDArray<float> x('c', {4, 128, 128, 4});
    NDArray<float> w('c', {4, 5, 4});
    NDArray<float> exp('c', {4, 64, 43, 4});


    nd4j::ops::dilation2d<float> op;
    auto result = op.execute({&x, &w}, {}, {1, 1,5,7,1, 1,2,3,1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}


TEST_F(DeclarableOpsTests6, Test_Dilation2D_Again_2) {
    NDArray<float> x('c', {4, 26, 19, 4});
    NDArray<float> w('c', {11, 7, 4});

    nd4j::ops::dilation2d<float> op;
    auto result = op.execute({&x, &w}, {}, {0, 1,2,3,1, 1,3,2,1});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_Conv3D_NDHWC_11) {
    NDArray<float> x('c', {4, 2, 28, 28, 3});
    NDArray<float> w('c', {2, 5, 5, 3, 4});
    NDArray<float> exp('c', {4, 1, 7, 10, 4});

    nd4j::ops::conv3dNew<float> op;
    auto result = op.execute({&x, &w}, {}, {2,5,5, 5,4,3, 0,0,0, 1,1,1, 0,0});
    ASSERT_EQ(Status::OK(), result->status());

    ShapeList shapeList({x.shapeInfo(), w.shapeInfo()});
    ContextPrototype<float> proto;
    Context<float> ctx(1);
    ctx.getIArguments()->push_back(2);
    ctx.getIArguments()->push_back(5);
    ctx.getIArguments()->push_back(5);

    ctx.getIArguments()->push_back(5);
    ctx.getIArguments()->push_back(4);
    ctx.getIArguments()->push_back(3);

    ctx.getIArguments()->push_back(0);
    ctx.getIArguments()->push_back(0);
    ctx.getIArguments()->push_back(0);

    ctx.getIArguments()->push_back(1);
    ctx.getIArguments()->push_back(1);
    ctx.getIArguments()->push_back(1);

    ctx.getIArguments()->push_back(0);
    ctx.getIArguments()->push_back(0);

    auto shapes = op.calculateOutputShape(&shapeList, ctx);
    ASSERT_EQ(1, shapes->size());

    auto s = shapes->at(0);

    shape::printShapeInfoLinear("calculated shape", s);

    auto z = result->at(0);
    z->printShapeInfo("z shape");

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;

    shapes->destroy();
    delete shapes;
}
