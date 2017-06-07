//
// Created by agibsonccc on 1/6/17.
//
#include "testinclude.h"


class SetElementWiseStrideTest :  public testing::Test {
public:
    int shape[3] = {8,4,8};
};






TEST_F(SetElementWiseStrideTest,SetTest) {
    int *shapeBufferF = shape::shapeBuffer(3,shape);
    const int N = 256;
    float *data = new float[N];
    for(int i = 0; i < N; i++) {
        data[i] = i + 1;
    }

    NativeOps *ops = new NativeOps();
    float *result = new float[N];
    ops->execTransformFloat(nullptr,63,data,1,result,1,nullptr,256);
    for(int i = 0; i < N; i++) {
        ASSERT_FLOAT_EQ(data[i],result[i]);
    }


    delete[] shapeBufferF;
    delete[] data;
    delete[] result;

}

