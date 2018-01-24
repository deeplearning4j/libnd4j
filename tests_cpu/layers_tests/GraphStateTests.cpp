//
//  @author raver119@gmail.com
//

#include "testlayers.h"
#include <graph/GraphState.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/LegacyTransformOp.h>
#include <NativeOps.h>

using namespace nd4j;
using namespace nd4j::graph;

class GraphStateTests : public testing::Test {
public:

};

/*
 * PLAN:
 * Create GraphState
 * Register Scope
 * Add few Ops to it
 * Call conditional, that refers to scopes 
 * Check results
 */

TEST_F(GraphStateTests, Basic_Tests_1) {
    NativeOps nativeOps;

    auto state = nativeOps.getGraphStateFloat(117L);
    ASSERT_EQ(117L, state->id());

    // this call will create scope internally
    state->registerScope(119);

    nd4j::ops::add<float> opA;
    nd4j::ops::LegacyTransformOp<float> opB(6); // simdOps::Neg

    ArgumentsList argsA;
    ArgumentsList argsB;

    state->attachOpToScope(119, &opA, argsA);
    state->attachOpToScope(119, &opB, argsB);

    auto scope = state->getScope(119);
    ASSERT_TRUE(scope != nullptr);
    ASSERT_EQ(2, scope->size());    

    nativeOps.deleteGraphStateFloat(state);
}

// just separate case for doubles wrapper in NativeOps, nothing else
TEST_F(GraphStateTests, Basic_Tests_2) {
    NativeOps nativeOps;

    auto state = nativeOps.getGraphStateDouble(117L);
    ASSERT_EQ(117L, state->id());

    // this call will create scope internally
    state->registerScope(119);

    nd4j::ops::add<double> opA;
    nd4j::ops::LegacyTransformOp<double> opB(6); // simdOps::Neg

    ArgumentsList argsA;
    ArgumentsList argsB;

    state->attachOpToScope(119, &opA, argsA);
    state->attachOpToScope(119, &opB, argsB);

    auto scope = state->getScope(119);
    ASSERT_TRUE(scope != nullptr);
    ASSERT_EQ(2, scope->size());    

    nativeOps.deleteGraphStateDouble(state);
}

TEST_F(GraphStateTests, Stateful_Execution_1) {
    NativeOps nativeOps;
}