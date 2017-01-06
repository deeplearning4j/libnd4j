//
// Created by agibsonccc on 1/6/17.
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <shape.h>


namespace {
    class TADTest : public testing::Test {

    };
}



TEST_F(TADTest,test1) {
    int shapeTest[2] = {2,2};
    int *shapeBuffer = shape::shapeBuffer(2,shapeTest);
    ASSERT_EQ("","");
    delete[] shapeBuffer;
}