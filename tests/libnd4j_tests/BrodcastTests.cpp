//
// Created by agibsonccc on 1/19/17.
//

#include "testinclude.h"
#include <broadcasting.h>

class BroadcastMultiDimTest : public testing::Test {
public:
    int dimensions[2] = {0,2};
    int inputShapeBuffer[8] = {3,2,3,5,15,5,1,0,1,99};
    float data[30] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,0.0,0.0,21.0,22.0,23.0,0.0,0.0,26.0,27.0,28.0,0.0,0.0};
    float result[30] = {0.0};
    float broadcastData[10] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0};
    int broadcastShapeInfo[8] = {2,2,5,5,1,0,1,99};
    int opNum = 2;
};

TEST_F(BroadcastMultiDimTest,MultimDimTest) {
    shape::TAD *tad = new shape::TAD(inputShapeBuffer,dimensions,2);
    tad->createTadOnlyShapeInfo();
    tad-> createOffsets();


   functions::broadcast::Broadcast<float>::exec(data,
                                                inputShapeBuffer,
                                                broadcastData,
                                                broadcastShapeInfo,
                                                result,inputShapeBuffer,
                                                dimensions,
                                                2,tad->tadOnlyShapeInfo,
                                                tad->tadOffsets,
                                                tad->tadOnlyShapeInfo,
                                                tad->tadOffsets);
}