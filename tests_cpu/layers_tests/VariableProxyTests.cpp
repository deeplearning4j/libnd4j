//
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <graph/VariableProxy.h>

using namespace nd4j;
using namespace nd4j::graph;

class VariableProxyTests : public testing::Test {
public:

};


TEST_F(VariableProxyTests, Test_Simple_1) {
    auto x = new NDArray<float>('c', {2, 2}, {1, 2, 3, 4});
    VariableSpace<float> ref;

    ref.putVariable(119, x);

    ASSERT_TRUE(ref.hasVariable(119));

    VariableProxy<float> proxy(&ref);

    ASSERT_TRUE(proxy.hasVariable(119));
}


TEST_F(VariableProxyTests, Test_Simple_2) {
    auto x = new NDArray<float>('c', {2, 2}, {1, 2, 3, 4});
    VariableSpace<float> ref;

    ASSERT_FALSE(ref.hasVariable(119));

    VariableProxy<float> proxy(&ref);

    ASSERT_FALSE(proxy.hasVariable(119));

    proxy.putVariable(119, x);

    ASSERT_FALSE(ref.hasVariable(119));

    ASSERT_TRUE(proxy.hasVariable(119));
}