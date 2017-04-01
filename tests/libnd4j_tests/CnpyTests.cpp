//
// Created by agibsonccc on 3/30/17.
//

#include "testinclude.h"

class FileTest : public testing::Test {

};

class LoadFromStringTest :  public testing::Test {

};

TEST_F(FileTest,T) {
    cnpy::NpyArray npy = cnpy::npyLoad(std::string("/home/agibsonccc/code/libnd4j/test.npy"));
    ASSERT_FALSE(npy.fortranOrder);

    ASSERT_EQ(2,npy.shape[0]);
    ASSERT_EQ(2,npy.shape[1]);
}

TEST_F(LoadFromStringTest,PathTest) {
    char *loaded = cnpy::loadFile("/home/agibsonccc/code/libnd4j/test.npy");
    cnpy::NpyArray loadedArr = cnpy::loadNpyFromPointer(loaded);
    ASSERT_FALSE(loadedArr.fortranOrder);
    ASSERT_EQ(2,loadedArr.shape[0]);
    ASSERT_EQ(2,loadedArr.shape[1]);
    Nd4jPointer  pointer = reinterpret_cast<Nd4jPointer >(&loadedArr);
    NativeOps nativeOps;
    Nd4jPointer  pointer1 = nativeOps.dataPointForNumpy(pointer);
    cnpy::NpyArray *loadedArrPointer = reinterpret_cast<cnpy::NpyArray *>(pointer1);
    ASSERT_EQ(2,loadedArrPointer->shape[0]);
    ASSERT_EQ(2,loadedArrPointer->shape[1]);
    delete[] loaded;
}

