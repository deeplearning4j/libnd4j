//
// Created by agibsonccc on 1/6/17.
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <shape.h>
#include <data_gen.h>

namespace {
    class VectorTest : public testing::Test {

    };

    class ShapeTest :  public testing::Test {
    public:
        int vectorShape[2] = {1,2};
    };

    class MatrixTest : public testing::Test {
    public:
        int rows = 3;
        int cols = 4;
        int rank = 2;
        int dims[2] = {0,1};
    };

    class TensorOneDimTest : public testing::Test {
    public:
        int rows = 3;
        int cols = 4;
        int dim2 = 5;
        int rank = 3;
        int dims[3] = {0,1,2};
    };

    class TensorTwoDimTest : public testing::Test {
    public:
        //From a 3d array:
        int rows = 3;
        int cols = 4;
        int dim2 = 5;
        int dimensionLength = 2;
        int dims[3][2] = {
                {0,1},{0,2},{1,2}
        };
    };

    class TensorTwoFromFourDDimTest : public testing::Test {
    public:
        //From a 3d array:
        int rows = 3;
        int cols = 4;
        int dim2 = 5;
        int dim3 = 6;
        int dimensionLength = 2;
        //Along dimension 0,1: expect matrix with shape [rows,cols]
        //Along dimension 0,2: expect matrix with shape [rows,dim2]
        //Along dimension 0,3: expect matrix with shape [rows,dim3]
        //Along dimension 1,2: expect matrix with shape [cols,dim2]
        //Along dimension 1,3: expect matrix with shape [cols,dim3]
        //Along dimension 2,3: expect matrix with shape [dim2,dim3]

        int dims[6][2] = {
                {0,1},{0,2},{0,3},{1,2},{1,3},{2,3}
        };
    };



}

::testing::AssertionResult arrsEquals(int n,int *assertion,int *other) {
    for(int i = 0; i < n; i++) {
        if(assertion[i] != other[i])
            return ::testing::AssertionFailure()  << i << "  is not equal";

    }
    return ::testing::AssertionSuccess();

}


TEST_F(TensorTwoFromFourDDimTest,TadTwoFromFourDimTest) {
    //Along dimension 0,1: expect matrix with shape [rows,cols]
    //Along dimension 0,2: expect matrix with shape [rows,dim2]
    //Along dimension 0,3: expect matrix with shape [rows,dim3]
    //Along dimension 1,2: expect matrix with shape [cols,dim2]
    //Along dimension 1,3: expect matrix with shape [cols,dim3]
    //Along dimension 2,3: expect matrix with shape [dim2,dim3]

    for(int i = 0; i <  3; i++) {
        int *dimArr = dims[i];


    }

}

TEST_F(TensorTwoDimTest,TadTwoDimTest) {
    //Along dimension 0,1: expect matrix with shape [rows,cols]
    //Along dimension 0,2: expect matrix with shape [rows,dim2]
    //Along dimension 1,2: expect matrix with shape [cols,dim2]
    for(int i = 0; i <  3; i++) {
        int *dimArr = dims[i];


    }

}

TEST_F(TensorOneDimTest,TadDimensionsForTensor) {
    int shape[3] = {rows,cols,dim2};
    int *shapeBuffer = shape::shapeBuffer(rank,shape);

    for(int i = 0; i < rank; i++) {
        int expectedDimZeroShape[2] = {1,shape[i]};
        int *expectedDimZeroShapeBuffer = shape::shapeBuffer(2,expectedDimZeroShape);
        //Along dimension 0: expect row vector with length 'dims[i]'
        shape::TAD *zero = new shape::TAD(shapeBuffer,&dims[i],1);
        int *testDimZeroShapeBuffer = zero->shapeInfoOnlyShapeAndStride();
        EXPECT_TRUE(arrsEquals(shape::shapeInfoLength(2),expectedDimZeroShape,testDimZeroShapeBuffer));
        delete[] testDimZeroShapeBuffer;
        delete[] expectedDimZeroShapeBuffer;

    }

    delete[] shapeBuffer;
}


TEST_F(MatrixTest,TadDimensionsForMatrix) {
    int shape[2] = {rows,cols};
    int *shapeBuffer = shape::shapeBuffer(rank,shape);
    shape::TAD *dimZero = new shape::TAD(shapeBuffer,&dims[0],1);
    shape::TAD *dimOne = new shape::TAD(shapeBuffer,&dims[1],1);
    //Along dimension 0: expect row vector with length 'rows'
    int rowVectorShape[2] = {1,rows};
    int *expectedDimZeroShape = shape::shapeBuffer(2,rowVectorShape);
    int *testDimZero = dimZero->shapeInfoOnlyShapeAndStride();
    EXPECT_TRUE(arrsEquals(shape::shapeInfoLength(2),expectedDimZeroShape,testDimZero));
    delete[] testDimZero;
    delete[] expectedDimZeroShape;
    //Along dimension 1: expect row vector with length 'cols'
    int rowVectorColShape[2] {1,cols};
    int *expectedDimOneShape = shape::shapeBuffer(2,rowVectorColShape);
    int *testDimOneShape = dimOne->shapeInfoOnlyShapeAndStride();
    EXPECT_TRUE(arrsEquals(shape::shapeInfoLength(2),expectedDimOneShape,testDimOneShape));
    delete[] testDimOneShape;
    delete[] expectedDimOneShape;
    delete dimOne;
    delete dimZero;
    delete[] shapeBuffer;
}

TEST_F(VectorTest,VectorTadShape) {
    int rowVector[2] = {2,2};
    int *rowBuffer = shape::shapeBuffer(2,rowVector);
    int rowDimension = 1;

    int columnVector[2] = {2,2};
    int *colShapeBuffer = shape::shapeBuffer(2,columnVector);
    int colDimension = 0;


    shape::TAD *rowTad = new shape::TAD(rowBuffer,&rowDimension,1);
    int *rowTadShapeBuffer = rowTad->shapeInfoOnlyShapeAndStride();
    int *rowTadShape = shape::shapeOf(rowTadShapeBuffer);
    shape::TAD *colTad = new shape::TAD(colShapeBuffer,&colDimension,1);
    int *colTadShapeBuffer = colTad->shapeInfoOnlyShapeAndStride();
    int *colTadShape = shape::shapeOf(colTadShapeBuffer);
    int assertionShape[2] = {1,5};
    EXPECT_TRUE(arrsEquals(2,assertionShape,rowTadShape));
    EXPECT_TRUE(arrsEquals(2,assertionShape,colTadShape));

    delete[] rowBuffer;
    delete[] colShapeBuffer;
    delete rowTad;
    delete colTad;
}




TEST_F(ShapeTest,IsVector) {
    ASSERT_TRUE(shape::isVector(vectorShape,2));
}

TEST_F(VectorTest,LinspaceCombinationTest) {
    int rows = 3;
    int cols = 4;
    int len = rows * cols;
    double *linspaced = linspace<double>(1,rows * cols,len);
    int shape[2] = {rows,cols};
    int *shapeBuffer = shape::shapeBuffer(2,shape);


    delete[] shapeBuffer;
    delete[] linspaced;
}