//
// Created by raver on 5/19/2018.
//

#include "testlayers.h"
#include <pointercast.h>
#include <Nd4j.h>
#include <NDArray.h>

using namespace nd4j;
using namespace nd4j::ops;

class Nd4jFactoryTests : public testing::Test {

};

TEST_F(Nd4jFactoryTests,  Basic_Test_0) {
    auto o = Nd4j::o<float>();
    auto p = Nd4j::p<float>();

    ASSERT_NE(nullptr, o);
    ASSERT_NE(nullptr, p);
}


TEST_F(Nd4jFactoryTests,  Basic_Test_1) {
    auto o = Nd4j::o<float>();
    auto p = Nd4j::p<float>();

    LaunchContext ctx;

    auto o2 = Nd4j::o<float>(&ctx);
    auto p2 = Nd4j::p<float>(&ctx);

    ASSERT_NE(o2, o);
    ASSERT_NE(p2, p);

    ASSERT_NE(nullptr, o2);
    ASSERT_NE(nullptr, p2);
}

TEST_F(Nd4jFactoryTests,  Basic_Create_1) {
    auto arrayO = Nd4j::o<float>()->create({3, 3});
    auto arrayP = Nd4j::p<float>()->create({3, 3});

    ASSERT_EQ(9, arrayP->lengthOf());
    ASSERT_EQ(9, arrayO.lengthOf());

    ASSERT_TRUE(arrayP->isSameShape(arrayO));

    ASSERT_TRUE(arrayP->equalsTo(arrayO));
}