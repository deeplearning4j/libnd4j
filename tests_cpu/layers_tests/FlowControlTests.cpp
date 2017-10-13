//
// Created by raver119 on 13.10.2017.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::ops;
using namespace nd4j::graph;

class FlowControlTests : public testing::Test {
public:

};

TEST_F(FlowControlTests, SwitchTest1) {
    Graph<float> graph;

    auto variableSpace = graph.getVariableSpace();

    auto input = new NDArray<float>('c',{32, 100});
    input->assign(-119.0f);

    auto condtionX = new NDArray<float>('c', {1, 1});
    condtionX->putScalar(0, 0.0f);
    auto condtionY = new NDArray<float>('c', {1, 1});
    condtionY->putScalar(0, 0.0f);

    variableSpace->putVariable(-1, input);
    variableSpace->putVariable(-2, condtionX);
    variableSpace->putVariable(-3, condtionY);

    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 0, 2, {1}, {3});
    auto nodeCondition = new Node<float>(OpType_BOOLEAN, 0, 119, {-2, -3});

    nd4j::ops::eq_scalar<float> eqOp;
    nodeCondition->setCustomOp(&eqOp);

    auto nodeSwitch = new Node<float>(OpType_CUSTOM, 0, 3, {2, 119}, {4, 5});

    nd4j::ops::Switch<float> switchOp;
    nodeSwitch->setCustomOp(&switchOp);

    auto nodeZ0 = new Node<float>(OpType_TRANSFORM, 0, 4, {}, {});
    nodeZ0->pickInput(3, 0);
    auto nodeZ1 = new Node<float>(OpType_TRANSFORM, 0, 5, {}, {});
    nodeZ1->pickInput(3, 1);


    graph.addNode(nodeA);
    graph.addNode(nodeB);
    graph.addNode(nodeCondition);
    graph.addNode(nodeSwitch);
    graph.addNode(nodeZ0);
    graph.addNode(nodeZ1);

    graph.buildGraph();

    ASSERT_EQ(1, nodeZ0->input()->size());
    ASSERT_EQ(1, nodeZ1->input()->size());

    ASSERT_EQ(0, nodeA->getLayer());
    ASSERT_EQ(0, nodeCondition->getLayer());
    ASSERT_EQ(1, nodeB->getLayer());
    ASSERT_EQ(2, nodeSwitch->getLayer());
    ASSERT_EQ(3, nodeZ0->getLayer());
    ASSERT_EQ(3, nodeZ1->getLayer());


    Nd4jStatus status = GraphExecutioner<float>::execute(&graph);

    ASSERT_EQ(ND4J_STATUS_OK, status);
}

