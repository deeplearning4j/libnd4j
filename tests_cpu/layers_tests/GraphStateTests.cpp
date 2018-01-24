//
//  @author raver119@gmail.com
//

#include "testlayers.h"
#include <graph/GraphState.h>
#include <NativeOps.h>

using namespace nd4j;
using namespace nd4j::graph;

class GraphStateTests : public testing::Test {
public:

};

TEST_F(GraphStateTests, Basic_Tests_1) {
    /*
     * PLAN:
     * Create GraphState
     * Register Scope
     * Add few Ops to it
     * Call conditional, that refers to scopes 
     * Check results
     */
    
    NativeOps nativeOps;

    auto state = nativeOps.getGraphStateFloat(119L);
    ASSERT_EQ(119L, state->id());


    state->registerScope(119);



    nativeOps.deleteGraphStateFloat(state);
}