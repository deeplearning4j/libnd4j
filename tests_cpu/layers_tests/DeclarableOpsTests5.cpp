//
// @author raver119@gmail.com
//


#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>


using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests5 : public testing::Test {
public:

    DeclarableOpsTests5() {
        printf("\n");
        fflush(stdout);
    }
};


TEST_F(DeclarableOpsTests5, Test_PermuteEquality_1) {
    NDArray<float> x('c', {1, 60});
    NDArray<float> exp('c', {3, 5, 4}, {1.0, 6.0, 11.0, 16.0, 2.0, 7.0, 12.0, 17.0, 3.0, 8.0, 13.0, 18.0, 4.0, 9.0, 14.0, 19.0, 5.0, 10.0, 15.0, 20.0, 21.0, 26.0, 31.0, 36.0, 22.0, 27.0, 32.0, 37.0, 23.0, 28.0, 33.0, 38.0, 24.0, 29.0, 34.0, 39.0, 25.0, 30.0, 35.0, 40.0, 41.0, 46.0, 51.0, 56.0, 42.0, 47.0, 52.0, 57.0, 43.0, 48.0, 53.0, 58.0, 44.0, 49.0, 54.0, 59.0, 45.0, 50.0, 55.0, 60.0});
    NDArrayFactory<float>::linspace(1, x);
    x.reshapei('c', {3, 4, 5});

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x}, {}, {0, 2, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_PermuteEquality_0) {
    NDArray<float> x('c', {1, 60});
    NDArrayFactory<float>::linspace(1, x);
    NDArray<float> exp('c', {3, 4, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{0, 1, 2} shape");
//    x.printBuffer("{0, 1, 2} data");

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x}, {}, {0, 1, 2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_PermuteEquality_2) {
    NDArray<float> x('c', {1, 60});
    NDArrayFactory<float>::linspace(1, x);
    NDArray<float> exp('c', {4, 3, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 21.0, 22.0, 23.0, 24.0, 25.0, 41.0, 42.0, 43.0, 44.0, 45.0, 6.0, 7.0, 8.0, 9.0, 10.0, 26.0, 27.0, 28.0, 29.0, 30.0, 46.0, 47.0, 48.0, 49.0, 50.0, 11.0, 12.0, 13.0, 14.0, 15.0, 31.0, 32.0, 33.0, 34.0, 35.0, 51.0, 52.0, 53.0, 54.0, 55.0, 16.0, 17.0, 18.0, 19.0, 20.0, 36.0, 37.0, 38.0, 39.0, 40.0, 56.0, 57.0, 58.0, 59.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{1, 0, 2} shape");
//    x.printBuffer("{1, 0, 2} data");

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x}, {}, {1, 0, 2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_PermuteEquality_3) {
    NDArray<float> x('c', {1, 60});
    NDArrayFactory<float>::linspace(1, x);
    NDArray<float> exp('c', {4, 5, 3}, {1.0, 21.0, 41.0, 2.0, 22.0, 42.0, 3.0, 23.0, 43.0, 4.0, 24.0, 44.0, 5.0, 25.0, 45.0, 6.0, 26.0, 46.0, 7.0, 27.0, 47.0, 8.0, 28.0, 48.0, 9.0, 29.0, 49.0, 10.0, 30.0, 50.0, 11.0, 31.0, 51.0, 12.0, 32.0, 52.0, 13.0, 33.0, 53.0, 14.0, 34.0, 54.0, 15.0, 35.0, 55.0, 16.0, 36.0, 56.0, 17.0, 37.0, 57.0, 18.0, 38.0, 58.0, 19.0, 39.0, 59.0, 20.0, 40.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{1, 2, 0} shape");
//    x.printBuffer("{1, 2, 0} data");

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x}, {}, {1, 2, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_PermuteEquality_4) {
    NDArray<float> x('c', {1, 60});
    NDArrayFactory<float>::linspace(1, x);
    NDArray<float> exp('c', {5, 3, 4}, {1.0, 6.0, 11.0, 16.0, 21.0, 26.0, 31.0, 36.0, 41.0, 46.0, 51.0, 56.0, 2.0, 7.0, 12.0, 17.0, 22.0, 27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 3.0, 8.0, 13.0, 18.0, 23.0, 28.0, 33.0, 38.0, 43.0, 48.0, 53.0, 58.0, 4.0, 9.0, 14.0, 19.0, 24.0, 29.0, 34.0, 39.0, 44.0, 49.0, 54.0, 59.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{2, 0, 1} shape");
//    x.printBuffer("{2, 0, 1} data");

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x}, {}, {2, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_PermuteEquality_5) {
    NDArray<float> x('c', {1, 60});
    NDArrayFactory<float>::linspace(1, x);
    NDArray<float> exp('c', {5, 4, 3}, {1.0, 21.0, 41.0, 6.0, 26.0, 46.0, 11.0, 31.0, 51.0, 16.0, 36.0, 56.0, 2.0, 22.0, 42.0, 7.0, 27.0, 47.0, 12.0, 32.0, 52.0, 17.0, 37.0, 57.0, 3.0, 23.0, 43.0, 8.0, 28.0, 48.0, 13.0, 33.0, 53.0, 18.0, 38.0, 58.0, 4.0, 24.0, 44.0, 9.0, 29.0, 49.0, 14.0, 34.0, 54.0, 19.0, 39.0, 59.0, 5.0, 25.0, 45.0, 10.0, 30.0, 50.0, 15.0, 35.0, 55.0, 20.0, 40.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{2, 1, 0} shape");
//    x.printBuffer("{2, 1, 0} data");

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x}, {}, {2, 1, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_TTS_bp_1) {
    NDArray<float> x('c', {2, 1, 3});
    NDArray<float> eps('c', {2, 4, 3});


    nd4j::ops::tile_to_shape_bp<float> op;
    auto result = op.execute({&x, &eps}, {}, {2, 4, 3});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x.isSameShape(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_Rdiv_bp_1) {
    NDArray<float> x('c', {3, 1}, {1, 2, 3});
    NDArray<float> y('c', {1, 4}, {1, 2, 3, 4});
    NDArray<float> eps('c', {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});


    nd4j::ops::reversedivide<float> op_ff;
    auto result_ff = op_ff.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result_ff->status());

    auto z_ff = result_ff->at(0);
    ASSERT_TRUE(eps.isSameShape(z_ff));

    nd4j::ops::reversedivide_bp<float> op_bp;
    auto result_bp = op_bp.execute({&x, &y, &eps}, {}, {});
    ASSERT_EQ(Status::OK(), result_bp->status());

    auto z_bp = result_bp->at(0);
    ASSERT_TRUE(x.isSameShape(z_bp));

    delete result_ff;
    delete result_bp;
}


TEST_F(DeclarableOpsTests5, Test_Boolean_diff_1) {
    NDArray<float> x('c', {1, 1}, {1.0f});
    NDArray<float> y(2.0f);

    nd4j::ops::less<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_SpaceToBatch_1) {
    NDArray<float> x('c', {1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    NDArray<float> blocks('c', {2}, {2, 2});
    NDArray<float> paddings('c', {2, 2}, {0, 0, 0, 0});

    NDArray<float> exp('c', {4, 1, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    nd4j::ops::space_to_batch<float> op;
    auto result = op.execute({&x, &blocks, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_SpaceToBatch_2) {
    NDArray<float> x('c', {1, 2, 2, 1}, {1, 2, 3, 4});
    NDArray<float> blocks('c', {2}, {2, 2});
    NDArray<float> paddings('c', {2, 2}, {0, 0, 0, 0});

    NDArray<float> exp('c', {4, 1, 1, 1}, {1, 2, 3, 4});

    nd4j::ops::space_to_batch<float> op;
    auto result = op.execute({&x, &blocks, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_SpaceToBatch_3) {
    NDArray<float> x('c', {2, 2, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    NDArray<float> blocks('c', {2}, {2, 2});
    NDArray<float> paddings('c', {2, 2}, {0, 0, 2, 0});

    NDArray<float> exp('c', {8, 1, 3, 1}, {0, 1, 3, 0, 9, 11, 0, 2, 4, 0, 10, 12, 0, 5, 7, 0, 13, 15, 0, 6, 8, 0, 14, 16});

    nd4j::ops::space_to_batch<float> op;
    auto result = op.execute({&x, &blocks, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_BatchToSpace_1) {
    NDArray<float> x('c', {4, 1, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    NDArray<float> exp('c', {1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    NDArray<float> blocks('c', {2}, {2, 2});
    NDArray<float> crops('c', {2, 2}, {0, 0, 0, 0});

    nd4j::ops::batch_to_space<float> op;
    auto result = op.execute({&x, &blocks, &crops}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_BatchToSpace_2) {
    NDArray<float> x('c', {4, 1, 1, 1}, {1, 2, 3, 4});
    NDArray<float> exp('c', {1, 2, 2, 1}, {1, 2, 3, 4});
    NDArray<float> blocks('c', {2}, {2, 2});
    NDArray<float> crops('c', {2, 2}, {0, 0, 0, 0});

    nd4j::ops::batch_to_space<float> op;
    auto result = op.execute({&x, &blocks, &crops}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_BatchToSpace_3) {
    NDArray<float> x('c', {8, 1, 3, 1}, {0, 1, 3, 0, 9, 11, 0, 2, 4, 0, 10, 12, 0, 5, 7, 0, 13, 15, 0, 6, 8, 0, 14, 16});
    NDArray<float> exp('c', {2, 2, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    NDArray<float> blocks('c', {2}, {2, 2});
    NDArray<float> crops('c', {2, 2}, {0, 0, 2, 0});

    nd4j::ops::batch_to_space<float> op;
    auto result = op.execute({&x, &blocks, &crops}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_test1) {
    
    NDArray<float> expected('c', {3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});

    nd4j::ops::eye<float> op;
    ResultSet<float>* results = op.execute({&expected}, {}, {99, 3});
    NDArray<float>* output = results->at(0);
    // output->printIndexedBuffer();
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_test2) {
    
    NDArray<float> expected('c', {3, 4}, {1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0.});

    nd4j::ops::eye<float> op;
    ResultSet<float>* results = op.execute({&expected}, {}, {99, 3, 4});
    NDArray<float>* output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_test3) {
    
    NDArray<float> expected('c', {2, 3, 4}, {1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0., 1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0.});

    nd4j::ops::eye<float> op;
    ResultSet<float>* results = op.execute({&expected}, {}, {99, 3, 4, 2});
    NDArray<float>* output = results->at(0);
    // output->printIndexedBuffer();
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_test4) {
    
    NDArray<float> expected('c', {2, 2, 3, 4}, {1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0., 1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0., 1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0., 1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0.});

    nd4j::ops::eye<float> op;
    ResultSet<float>* results = op.execute({&expected}, {}, {99, 3, 4, 2, 2});
    NDArray<float>* output = results->at(0);
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test1) {
    
    NDArray<float> input('c', {3, 4, 5});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> seqLengths('c', {4}, {4,4,4,4});
    NDArray<float> exp('c', {3, 4, 5}, {4,  3,  2,  1,  5, 9,  8,  7,  6, 10, 14, 13, 12, 11, 15, 19, 18, 17, 16, 20, 24, 23, 22, 21, 25, 29, 28, 27, 26, 30, 34, 33, 32, 31, 35, 39, 38, 37, 36, 40, 44, 43, 42, 41, 45, 49, 48, 47, 46, 50, 54, 53, 52, 51, 55, 59, 58, 57, 56, 60});

    nd4j::ops::reverse_sequense<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {2, 1});
    NDArray<float>* output = results->at(0);
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test2) {
    
    NDArray<float> input('c', {3, 4, 5});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> seqLengths('c', {4}, {0,1,2,3});
    NDArray<float> exp('c', {3, 4, 5}, {1,  2,  3,  4,  5, 6,  7,  8,  9, 10, 12, 11, 13, 14, 15, 18, 17, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 31, 33, 34, 35, 38, 37, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 51, 53, 54, 55, 58, 57, 56, 59, 60});

    nd4j::ops::reverse_sequense<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {2, 1});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test3) {
    
    NDArray<float> input('c', {3, 4, 5});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> seqLengths('c', {3}, {2,3,4});
    NDArray<float> exp('c', {3, 4, 5}, {2,  1,  3,  4,  5, 7,  6,  8,  9, 10, 12, 11, 13, 14, 15, 17, 16, 18, 19, 20, 23, 22, 21, 24, 25, 28, 27, 26, 29, 30, 33, 32, 31, 34, 35, 38, 37, 36, 39, 40, 44, 43, 42, 41, 45, 49, 48, 47, 46, 50, 54, 53, 52, 51, 55, 59, 58, 57, 56, 60});

    nd4j::ops::reverse_sequense<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {2, 0});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test4) {
    
    NDArray<float> input('c', {3, 4, 5});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> seqLengths('c', {5}, {1, 2, 1, 2, 3});
    NDArray<float> exp('c', {3, 4, 5}, {1, 22,  3, 24, 45, 6, 27,  8, 29, 50, 11, 32, 13, 34, 55, 16, 37, 18, 39, 60, 21,  2, 23,  4, 25, 26,  7, 28,  9, 30, 31, 12, 33, 14, 35, 36, 17, 38, 19, 40, 41, 42, 43, 44,  5, 46, 47, 48, 49, 10, 51, 52, 53, 54, 15, 56, 57, 58, 59, 20});

    nd4j::ops::reverse_sequense<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {0, 2});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test5) {
    
    NDArray<float> input('c', {3, 4, 5});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> seqLengths('c', {5}, {1, 2, 4, 2, 3});
    NDArray<float> exp('c', {3, 4, 5}, {1,  7, 18,  9, 15, 6,  2, 13,  4, 10, 11, 12,  8, 14,  5, 16, 17,  3, 19, 20, 21, 27, 38, 29, 35, 26, 22, 33, 24, 30, 31, 32, 28, 34, 25, 36, 37, 23, 39, 40, 41, 47, 58, 49, 55, 46, 42, 53, 44, 50, 51, 52, 48, 54, 45, 56, 57, 43, 59, 60});

    nd4j::ops::reverse_sequense<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {1, 2});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test6) {
    
    NDArray<float> input('c', {3, 4, 5});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> seqLengths('c', {4}, {1, 2, 3, 2});
    NDArray<float> exp('c', {3, 4, 5}, {1,  2,  3,  4,  5, 26, 27, 28, 29, 30, 51, 52, 53, 54, 55, 36, 37, 38, 39, 40, 21, 22, 23, 24, 25, 6,  7,  8,  9, 10, 31, 32, 33, 34, 35, 16, 17, 18, 19, 20, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 11, 12, 13, 14, 15, 56, 57, 58, 59, 60});

    nd4j::ops::reverse_sequense<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {0, 1});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test7) {
    
    NDArray<float> input('c', {1, 5});
    NDArrayFactory<float>::linspace(1, input);
    std::vector<float> data = {3};
    NDArray<float> seqLengths('c', {1}, data);    
    NDArray<float> exp('c', {1, 5}, {3, 2, 1, 4, 5});

    nd4j::ops::reverse_sequense<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {1, 0});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test8) {
    
    NDArray<float> input('c', {1, 5});
    NDArrayFactory<float>::linspace(1, input);
    std::vector<float> data = {1,0,1,0,1};
    NDArray<float> seqLengths('c', {5}, data);    
    NDArray<float> exp('c', {1, 5}, {1, 2, 3, 4, 5});

    nd4j::ops::reverse_sequense<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {0, 1});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test9) {
    
    NDArray<float> input('c', {5, 1});
    NDArrayFactory<float>::linspace(1, input);
    std::vector<float> data = {1,0,1,0,1};
    NDArray<float> seqLengths('c', {5}, data);    
    NDArray<float> exp('c', {5, 1}, {1, 2, 3, 4, 5});

    nd4j::ops::reverse_sequense<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {1, 0});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test10) {
    
    NDArray<float> input('c', {5, 1});
    NDArrayFactory<float>::linspace(1, input);
    std::vector<float> data = {3};
    NDArray<float> seqLengths('c', {1}, data);    
    NDArray<float> exp('c', {5, 1}, {3, 2, 1, 4, 5});

    nd4j::ops::reverse_sequense<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {0, 1});
    NDArray<float>* output = results->at(0);        
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}
