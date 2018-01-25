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

    auto state = nativeOps.getGraphStateFloat(117L);

    Nd4jIndex scopes[] = {22, 33};
    auto status = nativeOps.execCustomOpWithScopeFloat(nullptr, state, 10, scopes, 2, nullptr, nullptr, 0, nullptr, nullptr, 0);
    ASSERT_EQ(Status::THROW(), status);

    nativeOps.deleteGraphStateFloat(state);
}

TEST_F(GraphStateTests, Stateful_Execution_2) {
    NativeOps nativeOps;

    auto state = nativeOps.getGraphStateFloat(117L);

    state->registerScope(22);
    state->registerScope(33);

    Nd4jIndex scopes[] = {22, 33};
    auto status = nativeOps.execCustomOpWithScopeFloat(nullptr, state, 10, scopes, 2, nullptr, nullptr, 0, nullptr, nullptr, 0);
    
    // it's no-op: just LogicScope
    ASSERT_EQ(Status::OK(), status);

    nativeOps.deleteGraphStateFloat(state);
}


TEST_F(GraphStateTests, Stateful_Execution_3) {
    NativeOps nativeOps;

    NDArray<float> var0('c', {2, 2}, {1, 2, 3, 4});
    NDArray<float> var1(1.0f);
    NDArray<float> var2(1.0f);

    NDArray<float> res0('c', {2, 2});
    NDArray<float> res1(0.0f);
    NDArray<float> res2(0.0f);

    auto state = nativeOps.getGraphStateFloat(117L);

    Nd4jPointer ptrBuffers[] = {(Nd4jPointer) var0.buffer(), (Nd4jPointer) var1.buffer(), (Nd4jPointer)var2.buffer()};
    Nd4jPointer ptrShapes[] = {(Nd4jPointer) var0.shapeInfo(), (Nd4jPointer) var1.shapeInfo(), (Nd4jPointer)var2.shapeInfo()};

    Nd4jPointer outBuffers[] = {(Nd4jPointer) res0.buffer(), (Nd4jPointer) res1.buffer(), (Nd4jPointer) res2.buffer()};
    Nd4jPointer outShapes[] = {(Nd4jPointer) res0.shapeInfo(), (Nd4jPointer) res1.shapeInfo(), (Nd4jPointer) res2.shapeInfo()};

    // conditional scope
    state->registerScope(22);

    // body scope
    state->registerScope(33);

    Nd4jIndex scopes[] = {22, 33};

    // we're executing while loop
    auto status = nativeOps.execCustomOpWithScopeFloat(nullptr, state, 0, scopes, 2, ptrBuffers, ptrShapes, 3, outBuffers, outShapes, 3);
    ASSERT_EQ(Status::OK(), status);

    nativeOps.deleteGraphStateFloat(state);
}