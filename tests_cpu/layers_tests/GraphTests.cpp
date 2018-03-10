//
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <flatbuffers/flatbuffers.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/Node.h>
#include <graph/Graph.h>
#include <graph/GraphUtils.h>
#include <NDArray.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/generic/parity_ops.cpp>

using namespace nd4j;
using namespace nd4j::graph;

class GraphTests : public testing::Test {
public:
    /*
    int cShape[] = {2, 2, 2, 2, 1, 0, 1, 99};
    int fShape[] = {2, 2, 2, 1, 2, 0, 1, 102};
     */
};

TEST_F(GraphTests, SingleInput1) {
    auto graph = new Graph<float>();

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    graph->getVariableSpace()->putVariable(-1, x);

    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 2, 2, {1}, {3});
    auto nodeC = new Node<float>(OpType_TRANSFORM, 0, 3, {2}, {});

    graph->addNode(nodeA);
    graph->addNode(nodeB);
    graph->addNode(nodeC);

    ASSERT_EQ(1, graph->rootNodes());
    ASSERT_EQ(3, graph->totalNodes());

    GraphExecutioner<float>::execute(graph);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(3));

    auto node3 = graph->getVariableSpace()->getVariable(3)->getNDArray();

    ASSERT_NEAR(0.4161468, node3->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete graph;
}

TEST_F(GraphTests, DoubleInput1) {
    auto graph = new Graph<float>();

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    auto y = new NDArray<float>('c', {5, 5});
    y->assign(-1.0);

    auto z = new NDArray<float>('c', {5, 5});

    graph->getVariableSpace()->putVariable(-1, x);
    graph->getVariableSpace()->putVariable(-2, y);
    graph->getVariableSpace()->putVariable(-3, z);

    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {3});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 0, 2, {-2}, {3});
    auto nodeC = new Node<float>(OpType_PAIRWISE, 0, 3, {1, 2}, {-3});

    graph->addNode(nodeA);
    graph->addNode(nodeB);
    graph->addNode(nodeC);

    ASSERT_EQ(2, graph->rootNodes());
    ASSERT_EQ(3, graph->totalNodes());

    GraphExecutioner<float>::execute(graph);

    ASSERT_NEAR(3.0, z->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete graph;
}

TEST_F(GraphTests, SingleInput3) {
    auto graph = new Graph<float>();

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    auto v0 = new NDArray<float>('c', {5, 5});
    auto v1 = new NDArray<float>('c', {5, 5});

    graph->getVariableSpace()->putVariable(-1, x);
    graph->getVariableSpace()->putVariable(-2, v0);
    graph->getVariableSpace()->putVariable(-3, v1);

    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2, 3});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 14, 2, {1}, {-2});
    auto nodeC = new Node<float>(OpType_TRANSFORM, 26, 3, {1}, {-3});

    graph->addNode(nodeA);
    graph->addNode(nodeB);
    graph->addNode(nodeC);

    ASSERT_EQ(1, graph->rootNodes());
    ASSERT_EQ(3, graph->totalNodes());

    GraphExecutioner<float>::execute(graph);

    ASSERT_NEAR(1.4142135, v0->reduceNumber<simdOps::Mean<float>>(), 1e-5);
    ASSERT_NEAR(1.0, v1->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete graph;
}

TEST_F(GraphTests, SingleInput4) {
    auto graph = new Graph<float>();

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    auto v0 = new NDArray<float>('c', {5, 5});
    auto v1 = new NDArray<float>('c', {5, 5});

    graph->getVariableSpace()->putVariable(-1, x);
    graph->getVariableSpace()->putVariable(-2, v0);
    graph->getVariableSpace()->putVariable(-3, v1);

    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 14, 2, {1}, {3});
    auto nodeC = new Node<float>(OpType_TRANSFORM, 6, 3, {2}, {4, 5});

    auto nodeS = new Node<float>(OpType_TRANSFORM, 26, 4, {3}, {-2});
    auto nodeE = new Node<float>(OpType_TRANSFORM, 27, 5, {3}, {-3});

    graph->addNode(nodeA);
    graph->addNode(nodeB);
    graph->addNode(nodeC);
    graph->addNode(nodeS);
    graph->addNode(nodeE);

    ASSERT_EQ(1, graph->rootNodes());
    ASSERT_EQ(5, graph->totalNodes());

    GraphExecutioner<float>::execute(graph);

    ASSERT_NEAR(1.0, v0->reduceNumber<simdOps::Mean<float>>(), 1e-5);
    ASSERT_NEAR(-1.4142135, v1->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete graph;
}


TEST_F(GraphTests, DoubleInput2) {
    auto graph = new Graph<float>();

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    auto y = new NDArray<float>('c', {5, 5});
    y->assign(-1.0);

    auto z0 = new NDArray<float>('c', {5, 5});
    auto z1 = new NDArray<float>('c', {5, 5});

    graph->getVariableSpace()->putVariable(-1, x);
    graph->getVariableSpace()->putVariable(-2, y);
    graph->getVariableSpace()->putVariable(-3, z0);
    graph->getVariableSpace()->putVariable(-4, z1);


    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 14, 2, {1}, {3});
    auto nodeC = new Node<float>(OpType_TRANSFORM, 6, 3, {2}, {-3});

    auto nodeT = new Node<float>(OpType_TRANSFORM, 0, 11, {-2}, {12});
    auto nodeU = new Node<float>(OpType_TRANSFORM, 14, 12, {11}, {13});
    auto nodeV = new Node<float>(OpType_TRANSFORM, 6, 13, {12}, {-4});

    graph->addNode(nodeA);
    graph->addNode(nodeB);
    graph->addNode(nodeC);
    graph->addNode(nodeT);
    graph->addNode(nodeU);
    graph->addNode(nodeV);

    ASSERT_EQ(2, graph->rootNodes());
    ASSERT_EQ(6, graph->totalNodes());

    GraphExecutioner<float>::execute(graph);

    ASSERT_NEAR(-1.4142135, z0->reduceNumber<simdOps::Mean<float>>(), 1e-5);
    ASSERT_NEAR(-1.0, z1->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete graph;
}


TEST_F(GraphTests, DoubleInput3) {
    auto graph = new Graph<float>();

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    auto y = new NDArray<float>('c', {5, 5});
    y->assign(-1.0);

    auto z0 = new NDArray<float>('c', {5, 5});
    auto z1 = new NDArray<float>('c', {5, 5});


    auto w = new NDArray<float>('c', {5, 5});

    graph->getVariableSpace()->putVariable(-1, x);
    graph->getVariableSpace()->putVariable(-2, y);
    graph->getVariableSpace()->putVariable(-3, z0);
    graph->getVariableSpace()->putVariable(-4, z1);
    graph->getVariableSpace()->putVariable(-5, w);


    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 14, 2, {1}, {3});
    auto nodeC = new Node<float>(OpType_TRANSFORM, 6, 3, {2}, {-3, 21});

    auto nodeT = new Node<float>(OpType_TRANSFORM, 0, 11, {-2}, {12});
    auto nodeU = new Node<float>(OpType_TRANSFORM, 14, 12, {11}, {13});
    auto nodeV = new Node<float>(OpType_TRANSFORM, 6, 13, {12}, {-4, 21});

    auto nodeW = new Node<float>(OpType_PAIRWISE, 0, 21, {3, 13}, {22});
    auto nodeZ = new Node<float>(OpType_TRANSFORM, 0, 22, {21}, {-5});

    graph->addNode(nodeA);
    graph->addNode(nodeB);
    graph->addNode(nodeC);
    graph->addNode(nodeT);
    graph->addNode(nodeU);
    graph->addNode(nodeV);
    graph->addNode(nodeW);
    graph->addNode(nodeZ);

    ASSERT_EQ(2, graph->rootNodes());
    ASSERT_EQ(8, graph->totalNodes());

    GraphExecutioner<float>::execute(graph);

    ASSERT_NEAR(-1.4142135, z0->reduceNumber<simdOps::Mean<float>>(), 1e-5);
    ASSERT_NEAR(-1.0, z1->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    ASSERT_NEAR(2.4142135, w->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete graph;
}


TEST_F(GraphTests, QuadInput1) {
    auto graph = new Graph<float>();

    auto x0 = new NDArray<float>('c', {5, 5});
    x0->assign(0.0);

    auto x1 = new NDArray<float>('c', {5, 5});
    x1->assign(-1.0);

    auto x2 = new NDArray<float>('c', {5, 5});
    x2->assign(-2.0);

    auto x3 = new NDArray<float>('c', {5, 5});
    x3->assign(-3.0);

    auto z = new NDArray<float>('c', {5, 5});
    z->assign(119.0);

    graph->getVariableSpace()->putVariable(-1, x0);
    graph->getVariableSpace()->putVariable(-2, x1);
    graph->getVariableSpace()->putVariable(-3, x2);
    graph->getVariableSpace()->putVariable(-4, x3);
    graph->getVariableSpace()->putVariable(-5, z);

    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {11});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 0, 2, {-2}, {11});
    auto nodeC = new Node<float>(OpType_TRANSFORM, 0, 3, {-3}, {21});
    auto nodeD = new Node<float>(OpType_TRANSFORM, 0, 4, {-4}, {21});

    auto nodeP1 = new Node<float>(OpType_PAIRWISE, 0, 11, {1, 2}, {31});
    auto nodeP2 = new Node<float>(OpType_PAIRWISE, 0, 21, {3, 4}, {31});

    auto nodeZ = new Node<float>(OpType_PAIRWISE, 0, 31, {11, 21}, {-5});

    graph->addNode(nodeA);
    graph->addNode(nodeB);
    graph->addNode(nodeC);
    graph->addNode(nodeD);
    graph->addNode(nodeP1);
    graph->addNode(nodeP2);
    graph->addNode(nodeZ);

    ASSERT_EQ(4, graph->rootNodes());
    ASSERT_EQ(7, graph->totalNodes());

    GraphExecutioner<float>::execute(graph);

    ASSERT_NEAR(6.0, z->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete graph;
}

TEST_F(GraphTests, InternalBranching1) {
    auto graph = new Graph<float>();

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(0.0);

    auto z = new NDArray<float>('c', {5, 5});

    graph->getVariableSpace()->putVariable(-1, x);
    graph->getVariableSpace()->putVariable(-2, z);

    // 1.0
    auto nodeA = new Node<float>(OpType_TRANSFORM, 26, 1, {-1}, {11, 21});

    // -1
    auto nodeK = new Node<float>(OpType_TRANSFORM, 6, 11, {1}, {12});

    // 2.0
    auto nodeL = new Node<float>(OpType_TRANSFORM, 35, 12, {11}, {31});

    // -1
    auto nodeR = new Node<float>(OpType_TRANSFORM, 6, 21, {1}, {22});

    // 1
    auto nodeS = new Node<float>(OpType_TRANSFORM, 6, 22, {21}, {31});

    // 1.0
    auto nodeZ = new Node<float>(OpType_PAIRWISE, 0, 31, {12, 22}, {-2});

    graph->addNode(nodeA);
    graph->addNode(nodeK);
    graph->addNode(nodeL);
    graph->addNode(nodeR);
    graph->addNode(nodeS);
    graph->addNode(nodeZ);

    ASSERT_EQ(1, graph->rootNodes());
    ASSERT_EQ(6, graph->totalNodes());

    GraphExecutioner<float>::execute(graph);

    ASSERT_EQ(3, nodeZ->getLayer());

    ASSERT_NEAR(3.0, z->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete graph;
}


TEST_F(GraphTests, ReductionsTest1) {
    auto graph = new Graph<float>();

    auto x = new NDArray<float>('c', {5, 5});
    for (int r = 0; r < x->rows(); r++) {
        for (int c = 0; c < x->columns(); c++) {
            x->putScalar(r, c, -c);
        }
    }

    auto z = new NDArray<float>('c', {1, 5});

    graph->getVariableSpace()->putVariable(-1, x);
    graph->getVariableSpace()->putVariable(-2, z);


    auto nodeA = new Node<float>(OpType_ACCUMULATION, 0, 1, {-1}, {2}, {1});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 0, 2, {1}, {-2});

    graph->addNode(nodeA);
    graph->addNode(nodeB);

    ASSERT_EQ(1, graph->rootNodes());
    ASSERT_EQ(2, graph->totalNodes());

    GraphExecutioner<float>::execute(graph);

    ASSERT_NEAR(2.0, z->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete graph;
}


TEST_F(GraphTests, IndexReductionsTest1) {
    auto graph = new Graph<float>();

    auto x = new NDArray<float>('c', {5, 5});
    for (int r = 0; r < x->rows(); r++) {
        for (int c = 0; c < x->columns(); c++) {
            x->putScalar(r, c, -c);
        }
    }

    auto z = new NDArray<float>('c', {1, 5});

    graph->getVariableSpace()->putVariable(-1, x);
    graph->getVariableSpace()->putVariable(-2, z);


    auto nodeA = new Node<float>(OpType_INDEX_ACCUMULATION, 1, 1, {-1}, {2}, {1});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 0, 2, {1}, {-2});

    graph->addNode(nodeA);
    graph->addNode(nodeB);

    ASSERT_EQ(1, graph->rootNodes());
    ASSERT_EQ(2, graph->totalNodes());

    GraphExecutioner<float>::execute(graph);

    ASSERT_NEAR(4.0, z->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete graph;
}


TEST_F(GraphTests, AutoOutput1) {
    auto graph = new Graph<float>();
    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    graph->getVariableSpace()->putVariable(-1, x);

    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 35, 2, {1}, {});

    graph->addNode(nodeA);
    graph->addNode(nodeB);

    ASSERT_EQ(1, graph->rootNodes());
    ASSERT_EQ(2, graph->totalNodes());

    graph->buildGraph();

    ASSERT_TRUE(graph->getVariableSpace()->getVariable(2) != nullptr);

    GraphExecutioner<float>::execute(graph);

    auto outputs = graph->fetchOutputs();

    ASSERT_EQ(1, outputs->size());

    ASSERT_TRUE(outputs->at(0) != nullptr);

    ASSERT_NEAR(-1.0, outputs->at(0)->getNDArray()->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete outputs;
    delete graph;
}


TEST_F(GraphTests, AutoOutput2) {
    auto graph = new Graph<float>();
    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    graph->getVariableSpace()->putVariable(-1, x);

    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2, 3, -1});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 35, 2, {1}, {});
    auto nodeC = new Node<float>(OpType_TRANSFORM, 6, 3, {1}, {});

    graph->addNode(nodeA);
    graph->addNode(nodeB);
    graph->addNode(nodeC);

    ASSERT_EQ(1, graph->rootNodes());
    ASSERT_EQ(3, graph->totalNodes());

    graph->buildGraph();

    ASSERT_TRUE(graph->getVariableSpace()->getVariable(-1) != nullptr);
    ASSERT_TRUE(graph->getVariableSpace()->getVariable(2) != nullptr);
    ASSERT_TRUE(graph->getVariableSpace()->getVariable(3) != nullptr);

    GraphExecutioner<float>::execute(graph);

    auto outputs = graph->fetchOutputs();

    ASSERT_EQ(2, outputs->size());

    ASSERT_TRUE(outputs->at(0) != nullptr);

    ASSERT_NEAR(-1.0, outputs->at(0)->getNDArray()->reduceNumber<simdOps::Mean<float>>(), 1e-5);
    ASSERT_NEAR(-2.0, outputs->at(1)->getNDArray()->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete graph;
    delete outputs;
}


TEST_F(GraphTests, BroadcastTest1) {
    auto graph = new Graph<float>();
    auto x = new NDArray<float>('c', {5, 5});
    x->assign(0.0);

    auto y = new NDArray<float>('c', {1, 5});
    for (int e = 0; e < y->columns(); e++) {
        y->putScalar(e, e);
    }

    auto z = new NDArray<float>('c', {5, 5});

    graph->getVariableSpace()->putVariable(-1, x);
    graph->getVariableSpace()->putVariable(-2, y);
    graph->getVariableSpace()->putVariable(-3, z);

    auto nodeA = new Node<float>(OpType_BROADCAST, 0, 1, {-1, -2}, {2}, {1});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 6, 2, {1}, {-3});

    graph->addNode(nodeA);
    graph->addNode(nodeB);

    GraphExecutioner<float>::execute(graph);

    ASSERT_NEAR(-2.0, z->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete graph;
}


TEST_F(GraphTests, ScalarTest1) {
    auto graph = new Graph<float>();

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    auto z = new NDArray<float>('c', {5, 5});

    graph->getVariableSpace()->putVariable(-1, x);
    graph->getVariableSpace()->putVariable(-2, z);

    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 14, 2, {1}, {3});
    auto nodeE = new Node<float>(OpType_SCALAR, 0, 3, {2}, {-2}, {}, 1.3f);

    graph->addNode(nodeA);
    graph->addNode(nodeB);
    graph->addNode(nodeE);

    ASSERT_EQ(1, graph->rootNodes());
    ASSERT_EQ(3, graph->totalNodes());

    GraphExecutioner<float>::execute(graph);

    ASSERT_NEAR(2.714213, z->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete graph;
}

TEST_F(GraphTests, SymbolicLookupTest1) {
    auto graph = new Graph<float>();

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    auto z = new NDArray<float>('c', {5, 5});

    auto vX = new Variable<float>(x);
    auto vZ = new Variable<float>(z);

    std::string a("alpha");
    std::string o("omega");

    vX->setName(&a);
    vZ->setName(&o);

    graph->getVariableSpace()->putVariable(-1, vX);
    graph->getVariableSpace()->putVariable(-2, vZ);

    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 14, 2, {1}, {-2});

    std::string p("phi");
    std::string t("theta");

    nodeA->setName(&p);
    nodeB->setName(&t);

    graph->addNode(nodeA);
    graph->addNode(nodeB);


    auto rX = graph->getVariableSpace()->getVariable(&a);
    auto rZ = graph->getVariableSpace()->getVariable(&o);

    std::string om("omicron");

    ASSERT_TRUE(rX->getNDArray() == vX->getNDArray());
    ASSERT_TRUE(rZ->getNDArray() == vZ->getNDArray());
    ASSERT_FALSE(graph->getVariableSpace()->hasVariable(&om));


    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(1));
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(2));

    GraphExecutioner<float>::execute(graph);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(&p));
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(&t));

    ASSERT_NEAR(1.4142135, z->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete graph;
}

TEST_F(GraphTests, OutputValidation1) {
    auto graph = new Graph<float>();

    graph->getExecutorConfiguration()->_outputMode = OutputMode_EXPLICIT;

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    auto z = new NDArray<float>('c', {5, 5});

    auto vX = new Variable<float>(x);
    auto vZ = new Variable<float>(z);

    std::string a("alpha");
    std::string o("omega");

    vX->setName(&a);
    vZ->setName(&o);

    graph->getVariableSpace()->putVariable(-1, vX);
    graph->getVariableSpace()->putVariable(-2, vZ);

    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 14, 2, {1}, {-2});

    graph->addNode(nodeA);
    graph->addNode(nodeB);


    auto outputs = graph->fetchOutputs();

    ASSERT_EQ(0, outputs->size());

    delete graph;
    delete outputs;
}

TEST_F(GraphTests, OutputValidation2) {
    auto graph = new Graph<float>();

    graph->getExecutorConfiguration()->_outputMode = OutputMode_EXPLICIT;

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    auto z = new NDArray<float>('c', {5, 5});

    auto vX = new Variable<float>(x);
    auto vZ = new Variable<float>(z);

    std::string a("alpha");
    std::string o("omega");

    vX->setName(&a);
    vZ->setName(&o);

    graph->getVariableSpace()->putVariable(-1, vX);
    graph->getVariableSpace()->putVariable(-2, vZ);

    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 14, 2, {1}, {-2});

    graph->addNode(nodeA);
    graph->addNode(nodeB);

    graph->addOutput(-2);

    GraphExecutioner<float>::execute(graph);

    auto outputs = graph->fetchOutputs();

    ASSERT_EQ(1, outputs->size());

    ASSERT_NEAR(1.4142135, outputs->at(0)->getNDArray()->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete graph;
    delete outputs;
}

TEST_F(GraphTests, OutputValidation3) {
    auto graph = new Graph<float>();

    graph->getExecutorConfiguration()->_outputMode = OutputMode_IMPLICIT;

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    auto z = new NDArray<float>('c', {5, 5});

    auto vX = new Variable<float>(x);
    auto vZ = new Variable<float>(z);

    std::string a("alpha");
    std::string o("omega");

    vX->setName(&a);
    vZ->setName(&o);

    graph->getVariableSpace()->putVariable(-1, vX);
    graph->getVariableSpace()->putVariable(-2, vZ);

    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 14, 2, {1}, {});

    graph->addNode(nodeA);
    graph->addNode(nodeB);

    GraphExecutioner<float>::execute(graph);

    auto outputs = graph->fetchOutputs();

    ASSERT_EQ(1, outputs->size());

    ASSERT_NEAR(1.4142135, outputs->at(0)->getNDArray()->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete graph;
    delete outputs;
}

TEST_F(GraphTests, OutputValidation4) {
    auto graph = new Graph<float>();

    graph->getExecutorConfiguration()->_outputMode = OutputMode_EXPLICIT_AND_IMPLICIT;

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    auto z = new NDArray<float>('c', {5, 5});

    auto vX = new Variable<float>(x);
    auto vZ = new Variable<float>(z);

    std::string a("alpha");
    std::string o("omega");

    vX->setName(&a);
    vZ->setName(&o);

    graph->getVariableSpace()->putVariable(-1, vX);
    graph->getVariableSpace()->putVariable(-2, vZ);

    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 14, 2, {1}, {-2});

    graph->addOutput(-1);

    // not a typo. we want this value only once
    graph->addOutput(-1);

    graph->addNode(nodeA);
    graph->addNode(nodeB);

    GraphExecutioner<float>::execute(graph);

    auto outputs = graph->fetchOutputs();

    ASSERT_EQ(2, outputs->size());

    ASSERT_NEAR(1.4142135, outputs->at(1)->getNDArray()->reduceNumber<simdOps::Mean<float>>(), 1e-5);

    delete graph;
    delete outputs;
}


TEST_F(GraphTests, OutputValidation5) {
    auto graph = new Graph<float>();

    graph->getExecutorConfiguration()->_outputMode = OutputMode_VARIABLE_SPACE;

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    auto z = new NDArray<float>('c', {5, 5});

    auto vX = new Variable<float>(x);
    auto vZ = new Variable<float>(z);

    std::string a("alpha");
    std::string o("omega");

    vX->setName(&a);
    vZ->setName(&o);

    graph->getVariableSpace()->putVariable(-1, vX);
    graph->getVariableSpace()->putVariable(-2, vZ);

    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 14, 2, {1}, {-2});

    graph->addOutput(-1);

    graph->addNode(nodeA);
    graph->addNode(nodeB);

    GraphExecutioner<float>::execute(graph);

    auto outputs = graph->fetchOutputs();

    ASSERT_EQ(4, outputs->size());

    delete graph;
    delete outputs;
}

TEST_F(GraphTests, OutputValidation6) {
    auto graph = new Graph<float>();

    graph->getExecutorConfiguration()->_outputMode = OutputMode_VARIABLE_SPACE;

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    auto z = new NDArray<float>('c', {5, 5});

    auto vX = new Variable<float>(x);
    auto vZ = new Variable<float>(z);

    std::string a("alpha");
    std::string o("omega");

    vX->setName(&a);
    vZ->setName(&o);

    graph->getVariableSpace()->putVariable(-1, vX);
    graph->getVariableSpace()->putVariable(-2, vZ);

    auto nodeA = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeB = new Node<float>(OpType_TRANSFORM, 14, 2, {1}, {});

    //graph->addOutput(-1);

    graph->addNode(nodeA);
    graph->addNode(nodeB);

    GraphExecutioner<float>::execute(graph);

    auto outputs = graph->fetchOutputs();

//    nd4j_printf("Returned variables: \n", "");
//    for (int e = 0; e < outputs->size(); e++) {
//        printf("%i, ", outputs->at(e)->id());
//    }
//    printf("\n");

    ASSERT_EQ(4, outputs->size());

    //ASSERT_NEAR(1.4142135, graph->fetchOutputs()->at(1)->getNDArray()->reduceNumber<simdOps::Mean<float>>(), 1e-5);
    delete graph;
    delete outputs;
}

TEST_F(GraphTests, TestMultiOutput1) {
    auto graph = new Graph<float>();

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    auto y = new NDArray<float>('c', {5, 5});
    y->assign(-3.0);

    graph->getVariableSpace()->putVariable(-1, x);
    graph->getVariableSpace()->putVariable(-2, y);


    // Abs
    auto nodeA0 = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {11});
    nodeA0->markInplace(false);
    auto nodeB0 = new Node<float>(OpType_TRANSFORM, 0, 2, {-2}, {11});
    nodeB0->markInplace(false);

    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat("TestOp2i2o");

    // this op will add 1.0 to first input, and 2.0 for second input
    auto nodeT = new Node<float>(OpType_CUSTOM, 0, 11, {1, 2}, {21, 31}, {}, 0.0f);
    nodeT->setName("TestOp2i2o");
    nodeT->markInplace(false);
    nodeT->setCustomOp(op);


    // this op will subtract this value from 1.0
    auto nodeX = new Node<float>(OpType_TRANSFORM, 35, 21);
    nodeX->markInplace(false);
    nodeX->pickInput(11, 0);

    // this op will subtract this value from 1.0
    auto nodeY = new Node<float>(OpType_TRANSFORM, 35, 31);
    nodeY->markInplace(false);
    nodeY->pickInput(11, 1);

    graph->addNode(nodeA0);
    graph->addNode(nodeB0);
    graph->addNode(nodeT);
    graph->addNode(nodeX);
    graph->addNode(nodeY);

    std::pair<int, int> pair0(11,0);
    std::pair<int, int> pair1(11,1);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(pair0));
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(pair1));

    Nd4jStatus status = GraphExecutioner<float>::execute(graph);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_NEAR(-2.0f, graph->getVariableSpace()->getVariable(21)->getNDArray()->meanNumber(), 1e-5);
    ASSERT_NEAR(-4.0f, graph->getVariableSpace()->getVariable(31)->getNDArray()->meanNumber(), 1e-5);

    delete graph;
}

TEST_F(GraphTests, TestDivergentNode1) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat("Switch");
    auto nodeY = new Node<float>(OpType_CUSTOM, 0, 1);
    nodeY->setCustomOp(op);

    ASSERT_TRUE(nodeY->isDivergencePoint());
    ASSERT_TRUE(nodeY->isActive());

    delete nodeY;
}


TEST_F(GraphTests, MemoryEstimationTest1) {
    Graph<float> graph;

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    graph.getVariableSpace()->putVariable(-1, x);

    auto nodeA0 = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeA1 = new Node<float>(OpType_TRANSFORM, 0, 2, {1}, {});
    nodeA1->markInplace(false);

    graph.addNode(nodeA0);
    graph.addNode(nodeA1);

    ASSERT_EQ(2, graph.totalNodes());
    ASSERT_EQ(1, graph.rootNodes());

    auto memReq = graph.estimateRequiredMemory();

    ASSERT_EQ(25 * x->sizeOfT(), memReq);
}

TEST_F(GraphTests, MemoryEstimationTest2) {
    Graph<float> graph;

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    graph.getVariableSpace()->putVariable(-1, x);

    auto nodeA0 = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeA1 = new Node<float>(OpType_TRANSFORM, 0, 2, {1}, {});
    //nodeA1->markInplace(false);

    graph.addNode(nodeA0);
    graph.addNode(nodeA1);

    ASSERT_EQ(2, graph.totalNodes());
    ASSERT_EQ(1, graph.rootNodes());

    auto memReq = graph.estimateRequiredMemory();

    ASSERT_EQ(0, memReq);
}

TEST_F(GraphTests, MemoryEstimationTest3) {
    Graph<float> graph;

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    graph.getVariableSpace()->putVariable(-1, x);

    auto nodeA0 = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeA1 = new Node<float>(OpType_TRANSFORM, 0, 2, {1}, {3});
    auto nodeA2 = new Node<float>(OpType_ACCUMULATION, 0, 3, {2}, {}, {});
    nodeA1->markInplace(false);

    graph.addNode(nodeA0);
    graph.addNode(nodeA1);
    graph.addNode(nodeA2);

    ASSERT_EQ(3, graph.totalNodes());
    ASSERT_EQ(1, graph.rootNodes());

    auto memReq = graph.estimateRequiredMemory();

    ASSERT_EQ(26 * x->sizeOfT(), memReq);
}

TEST_F(GraphTests, MemoryEstimationTest4) {
    Graph<float> graph;

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    graph.getVariableSpace()->putVariable(-1, x);

    auto nodeA0 = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeA1 = new Node<float>(OpType_TRANSFORM, 0, 2, {1}, {3});
    auto nodeA2 = new Node<float>(OpType_ACCUMULATION, 0, 3, {2}, {}, {1});
    nodeA1->markInplace(false);

    graph.addNode(nodeA0);
    graph.addNode(nodeA1);
    graph.addNode(nodeA2);

    ASSERT_EQ(3, graph.totalNodes());
    ASSERT_EQ(1, graph.rootNodes());

    auto memReq = graph.estimateRequiredMemory();

    ASSERT_EQ(30 * x->sizeOfT(), memReq);
}

TEST_F(GraphTests, MemoryEstimationTest5) {
    Graph<float> graph;

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-2.0);

    graph.getVariableSpace()->putVariable(-1, x);

    auto nodeA0 = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeA1 = new Node<float>(OpType_TRANSFORM, 0, 2, {1}, {3});
    auto nodeA2 = new Node<float>(OpType_CUSTOM, 0, 3, {2}, {}, {});
    nodeA1->markInplace(false);

    nd4j::ops::testcustom<float> op;
    nodeA2->setCustomOp(&op);

    graph.addNode(nodeA0);
    graph.addNode(nodeA1);
    graph.addNode(nodeA2);

    graph.buildGraph();

    ASSERT_EQ(3, graph.totalNodes());
    ASSERT_EQ(1, graph.rootNodes());

    auto memReq = graph.estimateRequiredMemory();

    ASSERT_EQ((25 + 100) * x->sizeOfT(), memReq);
}

TEST_F(GraphTests, TestGraphInGraph_1) {
    // this one is external graph
    Graph<float> graphA;

    // and this ons is embedded
    Graph<float> graphB;

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-5.0);

    auto modifier = new NDArray<float>('c', {5, 5});
    modifier->assign(3.0);

    graphA.getVariableSpace()->putVariable(-1, x);
    graphB.getVariableSpace()->putVariable(-2, modifier);

    // this is placeholder variable
    graphB.getVariableSpace()->putVariable(-1, new Variable<float>(true));

    // abs, result is 5
    auto nodeA0 = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    // 1-, result -4
    auto nodeA1 = new Node<float>(OpType_TRANSFORM, 35, 2, {1}, {3});

    // graph should return 12: abs(3.0 x -4)
    auto nodeA2 = new Node<float>(OpType_GRAPH, -1, 3, {2}, {4});

    // 1 - 12 = -11
    auto nodeA3 = new Node<float>(OpType_TRANSFORM, 35, 4, {3}, {});

    nodeA2->setGraph(&graphB);

    graphA.addNode(nodeA0);
    graphA.addNode(nodeA1);
    graphA.addNode(nodeA2);
    graphA.addNode(nodeA3);

    // this is going to be PWT
    auto nodeB0 = new Node<float>(OpType_PAIRWISE, 6, 1, {-1, -2}, {2});
    auto nodeB1 = new Node<float>(OpType_TRANSFORM, 0, 2, {1}, {});

    graphB.addNode(nodeB0);
    graphB.addNode(nodeB1);

    graphB.buildGraph();
    graphA.buildGraph();

    ASSERT_EQ(0, nodeA0->getLayer());
    ASSERT_EQ(1, nodeA1->getLayer());
    ASSERT_EQ(2, nodeA2->getLayer());
    ASSERT_EQ(3, nodeA3->getLayer());

    ASSERT_EQ(0, nodeB0->getLayer());
    ASSERT_EQ(1, nodeB1->getLayer());

    Nd4jStatus status = GraphExecutioner<float>::execute(&graphA);
    ASSERT_EQ(ND4J_STATUS_OK, status);

    float m = graphA.getVariableSpace()->getVariable(4)->getNDArray()->meanNumber();

    //nd4j_printf("OpResult: %f\n", m);

    ASSERT_NEAR(-11.0, m, 1e-5);
}

// test for symbolic lookup
TEST_F(GraphTests, TestGraphInGraph_2) {
    // this one is external graph
    Graph<float> graphA;

    // and this ons is embedded
    Graph<float> graphB;

    auto x = new NDArray<float>('c', {5, 5});
    x->assign(-5.0);

    auto modifier = new NDArray<float>('c', {5, 5});
    modifier->assign(3.0);

    std::string nameA1("_nodeA1");
    
    graphA.getVariableSpace()->putVariable(-1, x);
    graphB.getVariableSpace()->putVariable(-2, modifier);

    // this is placeholder variable
    auto placeHolder = new Variable<float>(true);
    placeHolder->setName(&nameA1);
    graphB.getVariableSpace()->putVariable(-1, placeHolder);

    // abs, result is 5
    auto nodeA0 = new Node<float>(OpType_TRANSFORM, 0, 1, {-1}, {2});
    // 1-, result -4
    auto nodeA1 = new Node<float>(OpType_TRANSFORM, 35, 2, {1}, {3});
    nodeA1->setName(nameA1);

    // graph should return 12: abs(3.0 x -4)
    auto nodeA2 = new Node<float>(OpType_GRAPH, -1, 3, {2}, {4});

    // 1 - 12 = -11
    auto nodeA3 = new Node<float>(OpType_TRANSFORM, 35, 4, {3}, {});

    nodeA2->setGraph(&graphB);

    graphA.addNode(nodeA0);
    graphA.addNode(nodeA1);
    graphA.addNode(nodeA2);
    graphA.addNode(nodeA3);

    // this is going to be PWT
    auto nodeB0 = new Node<float>(OpType_PAIRWISE, 6, 1, {-1, -2}, {2});
    auto nodeB1 = new Node<float>(OpType_TRANSFORM, 0, 2, {1}, {});

    graphB.addNode(nodeB0);
    graphB.addNode(nodeB1);

    graphB.buildGraph();
    graphA.buildGraph();

    ASSERT_EQ(0, nodeA0->getLayer());
    ASSERT_EQ(1, nodeA1->getLayer());
    ASSERT_EQ(2, nodeA2->getLayer());
    ASSERT_EQ(3, nodeA3->getLayer());

    ASSERT_EQ(0, nodeB0->getLayer());
    ASSERT_EQ(1, nodeB1->getLayer());

    Nd4jStatus status = GraphExecutioner<float>::execute(&graphA);
    ASSERT_EQ(ND4J_STATUS_OK, status);

    float m = graphA.getVariableSpace()->getVariable(4)->getNDArray()->meanNumber();

    //nd4j_printf("OpResult: %f\n", m);

    ASSERT_NEAR(-11.0, m, 1e-5);
}


TEST_F(GraphTests, Test_Clone_1) {
    NDArray<float> exp('c', (std::vector<int>){3});
    exp.assign(3.0);


    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/reduce_dim.fb");
    auto variableSpace = graph->getVariableSpace();
    //graph->buildGraph();

    auto clone = graph->clone();

    Nd4jStatus statusOriginal = GraphExecutioner<float>::execute(graph);

    ASSERT_EQ(ND4J_STATUS_OK, statusOriginal);
    ASSERT_TRUE(variableSpace->hasVariable(3));

    Nd4jStatus statusClone = GraphExecutioner<float>::execute(clone);

    ASSERT_EQ(ND4J_STATUS_OK, statusClone);

    auto z0 = variableSpace->getVariable(4)->getNDArray();
    auto z1 = clone->getVariableSpace()->getVariable(4)->getNDArray();

    ASSERT_TRUE(exp.isSameShape(z0));
    ASSERT_TRUE(exp.equalsTo(z0));

    ASSERT_TRUE(exp.isSameShape(z1));
    ASSERT_TRUE(exp.equalsTo(z1));

    delete graph;
    delete clone;
}




TEST_F(GraphTests, Test_Clone_2) {
    NDArray<float> exp('c', (std::vector<int>){3});
    exp.assign(3.0);


    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/reduce_dim.fb");
    auto variableSpace = graph->getVariableSpace();
    graph->buildGraph();

    auto clone = graph->clone();

    Nd4jStatus statusOriginal = GraphExecutioner<float>::execute(graph);

    ASSERT_EQ(ND4J_STATUS_OK, statusOriginal);
    ASSERT_TRUE(variableSpace->hasVariable(3));

    Nd4jStatus statusClone = GraphExecutioner<float>::execute(clone);

    ASSERT_EQ(ND4J_STATUS_OK, statusClone);

    auto z0 = variableSpace->getVariable(4)->getNDArray();
    auto z1 = clone->getVariableSpace()->getVariable(4)->getNDArray();

    ASSERT_TRUE(exp.isSameShape(z0));
    ASSERT_TRUE(exp.equalsTo(z0));

    ASSERT_TRUE(exp.isSameShape(z1));
    ASSERT_TRUE(exp.equalsTo(z1));

    delete graph;
    delete clone;
}

TEST_F(GraphTests, Test_Dtype_Conversion_1) {
    NDArray<double> expD('c', {3}, {3.0, 3.0, 3.0});
    NDArray<float> expF('c', {3}, {3.0, 3.0, 3.0});

    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/reduce_dim.fb");
    graph->buildGraph();


    auto gd = graph->template asT<double>();
    auto gf = gd->template asT<float>();

    // checking float graph
    Nd4jStatus statusF = GraphExecutioner<float>::execute(gf);
    ASSERT_EQ(ND4J_STATUS_OK, statusF);

    ASSERT_TRUE(gf->getVariableSpace()->hasVariable(4));
    auto z1 = gf->getVariableSpace()->getVariable(4)->getNDArray();

    ASSERT_TRUE(expF.isSameShape(z1));
    ASSERT_TRUE(expF.equalsTo(z1));


    // checking double graph
    Nd4jStatus statusD = GraphExecutioner<double>::execute(gd);
    ASSERT_EQ(ND4J_STATUS_OK, statusD);

    ASSERT_TRUE(gd->getVariableSpace()->hasVariable(4));
    auto z2 = gd->getVariableSpace()->getVariable(4)->getNDArray();

    ASSERT_TRUE(expD.isSameShape(z2));
    ASSERT_TRUE(expD.equalsTo(z2));


    delete graph;
    delete gd;
    delete gf;
}

TEST_F(GraphTests, Test_Dtype_Conversion_2) {
    NDArray<float> expF('c', {5, 4}, {0.951276f, 0.501379f, 0.501368f, 0.968136f, -0.951359f, 0.499845f, -0.501381f, 0.976955f, -0.000073f, 0.499154f, 0.000098f, 0.972500f, -0.019765f, -0.499479f, -0.005979f, -0.965330f, 0.016531f, -0.500842f, 0.004861f, -0.965910f});
    NDArray<double> expD('c', {5, 4}, {0.951276f, 0.501379f, 0.501368f, 0.968136f, -0.951359f, 0.499845f, -0.501381f, 0.976955f, -0.000073f, 0.499154f, 0.000098f, 0.972500f, -0.019765f, -0.499479f, -0.005979f, -0.965330f, 0.016531f, -0.500842f, 0.004861f, -0.965910f});

    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/ae_00.fb");
    graph->buildGraph();


    auto gd = graph->template asT<double>();
    auto gf = gd->template asT<float>();

    // checking float
    auto resultF = GraphExecutioner<float>::execute(gf);
    ASSERT_EQ(ND4J_STATUS_OK, resultF);
    ASSERT_TRUE(gf->getVariableSpace()->hasVariable(18));
    auto zF = gf->getVariableSpace()->getVariable(18)->getNDArray();

    ASSERT_TRUE(expF.isSameShape(zF));
    ASSERT_TRUE(expF.equalsTo(zF));


    // checking double
    auto resultD = GraphExecutioner<double>::execute(gd);
    ASSERT_EQ(ND4J_STATUS_OK, resultD);
    ASSERT_TRUE(gd->getVariableSpace()->hasVariable(18));
    auto zD = gd->getVariableSpace()->getVariable(18)->getNDArray();

    ASSERT_TRUE(expD.isSameShape(zD));
    ASSERT_TRUE(expD.equalsTo(zD));

    delete graph;
    delete gd;
    delete gf;
}

TEST_F(GraphTests, Test_Hash_Function_1) {
    auto graph0 = GraphExecutioner<float>::importFromFlatBuffers("./resources/ae_00.fb");
    auto graph1 = GraphExecutioner<float>::importFromFlatBuffers("./resources/ae_00.fb");
    auto graph2 = GraphExecutioner<float>::importFromFlatBuffers("./resources/conv_0.fb");

    ASSERT_EQ(graph0->hashCode(), graph1->hashCode());
    ASSERT_NE(0L, graph1->hashCode());
    ASSERT_NE(graph0->hashCode(), graph2->hashCode());

    auto graph0D = graph0->template asT<double>();
    auto graph1D = graph1->template asT<double>();

    ASSERT_NE(graph0->hashCode(), graph0D->hashCode());
    ASSERT_EQ(graph0D->hashCode(), graph1D->hashCode());

    delete graph0;
    delete graph1;
    delete graph2;
    delete graph0D;
    delete graph1D;
}

TEST_F(GraphTests, OpListTest_1) {
    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/ae_00.fb"); ;

    ASSERT_TRUE(graph != nullptr);
    std::vector<OpDescriptor> ops = graph->getOperations();

    ASSERT_TRUE(ops.size() == 11);
    GraphUtils::filterOperations(ops);
    ASSERT_TRUE(ops.size() == 7);

    std::string exp("./buildnativeoperations.sh -g \"-D_rank;-D_range;-D_subtract;-D_transpose;-D_matmul;-D_biasadd;-D_TRANSFORM{15}\"");
    std::string out = GraphUtils::makeCommandLine(ops);
//    nd4j_printf("EXP: >%s<\n", exp.c_str());
//    nd4j_printf("OUT: >%s<\n", out.c_str());
    ASSERT_TRUE(0 == out.compare(exp));

    delete graph;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(GraphTests, OpListTest_2) {
    auto graph0 = GraphExecutioner<float>::importFromFlatBuffers("./resources/ae_00.fb");
    auto graph1 = GraphExecutioner<float>::importFromFlatBuffers("./resources/tensor_slice.fb");

    ASSERT_TRUE(graph0 != nullptr);
    ASSERT_TRUE(graph1 != nullptr);

    std::vector<OpDescriptor> ops = graph0->getOperations();
    std::vector<OpDescriptor> ops1 = graph1->getOperations();
    std::copy ( ops1.begin(), ops1.end(),  std::back_inserter(ops));

    ASSERT_TRUE(ops.size() == 13);

    GraphUtils::filterOperations(ops);

    std::string exp = "./buildnativeoperations.sh -g \"-D_rank;-D_range;-D_subtract;-D_transpose;-D_matmul;-D_biasadd;-D_TRANSFORM{15};-D_strided_slice;-D_ACCUMULATION{1}\"";

    ASSERT_TRUE(ops.size() == 9);
    ASSERT_EQ(exp, GraphUtils::makeCommandLine(ops));

    delete graph0;
    delete graph1;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(GraphTests, OpListTest_3) {
    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/ae_00.fb"); ;

    ASSERT_TRUE(graph != nullptr);
    std::vector<OpDescriptor> ops = graph->getOperations();
    std::vector<OpDescriptor> ops2(ops);
    std::copy(ops.begin(), ops.end(),  std::back_inserter(ops2));
    
    ASSERT_TRUE(ops.size() == 11);
    ASSERT_TRUE(ops2.size() == 2 * ops.size());

    GraphUtils::filterOperations(ops2);
    GraphUtils::filterOperations(ops);
    ASSERT_TRUE(ops.size() == ops2.size());
    ASSERT_TRUE(ops.size() == 7);
    ASSERT_TRUE(GraphUtils::makeCommandLine(ops) == GraphUtils::makeCommandLine(ops2));
    
    delete graph;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(GraphTests, OpListTest_4) {
    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/conv_0.fb"); ;

    ASSERT_TRUE(graph != nullptr);
    std::vector<OpDescriptor> ops = graph->getOperations();
    std::vector<OpDescriptor> ops2(ops);
    std::copy(ops.begin(), ops.end(),  std::back_inserter(ops2));

    nd4j_printf("Total ops before %i\n", ops.size());
    ASSERT_TRUE(ops.size() == 6);
    ASSERT_TRUE(ops2.size() == 2 * ops.size());

    GraphUtils::filterOperations(ops2);
    GraphUtils::filterOperations(ops);
    ASSERT_TRUE(ops.size() == ops2.size());
    ASSERT_TRUE(ops.size() == 5);
    ASSERT_TRUE(GraphUtils::makeCommandLine(ops) == GraphUtils::makeCommandLine(ops2));

    delete graph;
}
