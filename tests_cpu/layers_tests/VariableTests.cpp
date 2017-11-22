//
// @author raver119@gmail.com
//

#ifndef LIBND4J_VARIABLETESTS_H
#define LIBND4J_VARIABLETESTS_H

#include "testlayers.h"
#include <NDArray.h>
#include <graph/Variable.h>
#include <flatbuffers/flatbuffers.h>
#include <NDArrayFactory.h>

using namespace nd4j;
using namespace nd4j::graph;

class VariableTests : public testing::Test {
public:

};

TEST_F(VariableTests, TestClone_1) {
    auto array1 = new NDArray<float>(5,5, 'c');
    array1->assign(1.0);

    auto var1 = new Variable<float>(array1, "alpha");
    var1->setId(119);


    auto var2 = var1->clone();

    ASSERT_FALSE(var1->getNDArray() == var2->getNDArray());
    auto array2 = var2->getNDArray();

    ASSERT_TRUE(array1->equalsTo(array2));
    ASSERT_EQ(var1->id(), var2->id());
    ASSERT_EQ(*var1->getName(), *var2->getName());

    delete var1;

    std::string str("alpha");
    ASSERT_EQ(*var2->getName(), str);
    array2->assign(2.0);

    ASSERT_NEAR(2.0, array2->meanNumber(), 1e-5);

    delete var2;
}

TEST_F(VariableTests, Test_FlatVariableDataType_1) {
    flatbuffers::FlatBufferBuilder builder(1024);
    NDArray<float> original('c', {5, 10});
    NDArrayFactory<float>::linspace(1, original);

    auto vec = original.asByteVector();

    auto fShape = builder.CreateVector(original.getShapeInfoAsVector());
    auto fBuffer = builder.CreateVector(vec);

    auto fArray = CreateFlatArray(builder, fShape, fBuffer, nd4j::graph::DataType::DataType_FLOAT);

    auto flatVar = CreateFlatVariable(builder, 1, 0, 0, fArray);

    builder.Finish(flatVar);

    auto ptr = builder.GetBufferPointer();



    //auto restoredVar = new Variable<float>(flatVar);
}

#endif //LIBND4J_VARIABLETESTS_H
