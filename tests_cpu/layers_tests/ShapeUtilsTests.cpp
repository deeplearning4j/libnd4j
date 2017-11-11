//
// Created by raver119 on 01.11.2017.
//

#include "testlayers.h"
#include <helpers/ShapeUtils.h>


using namespace nd4j;
using namespace nd4j::graph;

class ShapeUtilsTests : public testing::Test {
public:

};

TEST_F(ShapeUtilsTests, BasicInject1) {
    std::vector<int> shape({1, 4});

    ShapeUtils<float>::insertDimension(2, shape.data(), -1, 3);
    ASSERT_EQ(4, shape.at(0));
    ASSERT_EQ(3, shape.at(1));
}


TEST_F(ShapeUtilsTests, BasicInject2) {
    std::vector<int> shape({1, 4});

    ShapeUtils<float>::insertDimension(2, shape.data(), 0, 3);
    ASSERT_EQ(3, shape.at(0));
    ASSERT_EQ(4, shape.at(1));
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, EvalBroadcastShapeInfo_1)
{

    int xShapeInfo[]   = {3, 3, 2, 2, 4, 2, 1, 0, 1, 99};
    int yShapeInfo[]   = {2,    1, 2,    2, 1, 0, 1, 99};
    int expShapeInfo[] = {3, 3, 2, 2, 4, 2, 1, 0, 1, 99};

    NDArray<float> x(xShapeInfo);
    NDArray<float> y(yShapeInfo);

    int *newShapeInfo = ShapeUtils<float>::evalBroadcastShapeInfo(x, y);
        
    ASSERT_TRUE(shape::equalsStrict(expShapeInfo, newShapeInfo));    

    RELEASE(newShapeInfo, x.getWorkspace());
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, EvalBroadcastShapeInfo_2)
{

    int xShapeInfo[]   = {4, 8, 1, 6, 1, 6,   6,  1, 1, 0, 1, 99};    
    int yShapeInfo[]   = {3,    7, 1, 5,      5,  5, 1, 0, 1, 99};
    int expShapeInfo[] = {4, 8, 7, 6, 5, 210, 30, 5, 1, 0, 1, 99};    

    NDArray<float> x(xShapeInfo);
    NDArray<float> y(yShapeInfo);

    int *newShapeInfo = ShapeUtils<float>::evalBroadcastShapeInfo(x, y);
        
    ASSERT_TRUE(shape::equalsStrict(expShapeInfo, newShapeInfo));    

    RELEASE(newShapeInfo, x.getWorkspace());
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, EvalBroadcastShapeInfo_3)
{

    int xShapeInfo[]   = {3, 15, 3, 5, 15, 5, 1, 0, 1, 99};
    int yShapeInfo[]   = {3, 15, 1, 5,  5, 5, 1, 0, 1, 99};
    int expShapeInfo[] = {3, 15, 3, 5, 15, 5, 1, 0, 1, 99};

    NDArray<float> x(xShapeInfo);
    NDArray<float> y(yShapeInfo);

    int *newShapeInfo = ShapeUtils<float>::evalBroadcastShapeInfo(x, y);
        
    ASSERT_TRUE(shape::equalsStrict(expShapeInfo, newShapeInfo));    

    RELEASE(newShapeInfo, x.getWorkspace());
}

// TEST_F(ShapeUtilsTests, EvalBroadcastShapeInfo_4)
// {

//     int xShapeInfo[]   = {2,    2, 1,     1, 1, 0, 1, 99};
//     int yShapeInfo[]   = {3, 8, 4, 3, 12, 3, 1, 0, 1, 99};
//     int expShapeInfo[] = {3, 15, 3, 5, 15, 5, 1, 0, 1, 99};

//     NDArray<float> x(xShapeInfo);
//     NDArray<float> y(yShapeInfo);

//     int *newShapeInfo = ShapeUtils<float>::evalBroadcastShapeInfo(x, y);

//     ASSERT_TRUE(shape::equalsStrict(expShapeInfo, newShapeInfo));

//     RELEASE(newShapeInfo, x.getWorkspace());
// }
