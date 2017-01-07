//
// Created by agibsonccc on 1/6/17.
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <shape.h>


namespace {
    class VectorTest : public testing::Test {

    };

    class ShapeTest :  public testing::Test {
    public:
        int vectorShape[2] = {1,2};
    };
}

::testing::AssertionResult arrsEquals(int n,int *assertion,int *other) {
    for(int i = 0; i < n; i++) {
        if(assertion[i] != other[i])
            return ::testing::AssertionFailure()  << i << "  is not equal";

    }
    return ::testing::AssertionSuccess();

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

