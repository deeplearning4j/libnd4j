//
// Created by raver119 on 29.11.17.
//


#include "testlayers.h"
#include <flatbuffers/flatbuffers.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/Node.h>
#include <graph/Graph.h>
#include <NDArray.h>
#include <ops/declarable/DeclarableOp.h>

using namespace nd4j;
using namespace nd4j::graph;

class GraphExecutionerTests : public testing::Test {
public:

};

TEST_F(GraphExecutionerTests, Test_Implicit_Output_1) {
    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/tensor_slice.fb");
    graph->buildGraph();

    auto outputs = graph->fetchOutputs();

    ASSERT_EQ(1, outputs->size());

    delete outputs;
    delete graph;
}