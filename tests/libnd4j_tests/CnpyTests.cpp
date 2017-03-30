//
// Created by agibsonccc on 3/30/17.
//

#include "testinclude.h"

class NpyTest : public testing::Test {

};


TEST_F(NpyTest,SixDWithOnes) {
   cnpy::NpyArray npy = cnpy::npyLoad(std::string("test.npy"));
}