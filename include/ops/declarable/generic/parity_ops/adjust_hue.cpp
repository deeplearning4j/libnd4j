//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/adjust_hue.h>

namespace nd4j {
namespace ops {
    CONFIGURABLE_OP_IMPL(adjust_hue, 1, 1, true, -2, -2) {
        auto input = INPUT_VARIABLE(0);

        REQUIRE_TRUE(input->rankOf() == 3 || input->rankOf() == 4, 0, "AdjustHue: op expects either 3D or 4D input, but got %i instead", input->rankOf());

        return ND4J_STATUS_OK;
    }
}
}