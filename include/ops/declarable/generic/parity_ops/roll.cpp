//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_roll)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/roll.h>

namespace nd4j {
namespace ops {

    CONFIGURABLE_OP_IMPL(roll, 1, 1, true, 0, 1) {

        NDArray<T>* output = OUTPUT_VARIABLE(0);
        NDArray<T>* input = INPUT_VARIABLE(0);
        bool shiftIsLinear = true;
        //std::vector<int> axes(input->rankOf());
        int shift = INT_ARG(0);
        if (block.numI() > 1)
            shiftIsLinear = false;
        if (shiftIsLinear) {
            helpers::rollFunctorLinear(input, output, shift);
        }
        else {
            std::vector<int> axes(block.numI() - 1);
            for (unsigned e = 0; e < axes.size(); ++e) {
                int axe = INT_ARG(e + 1);
                REQUIRE_TRUE(axe < input->rankOf() && axe >= -input->rankOf(), 0, "roll: axe value should be between -%i and %i, but %i was given.",
                    input->rankOf(), input->rankOf() - 1, axe);
                axes[e] = (axe < 0? (input->rankOf() + axe) : axe);
            }
            helpers::rollFunctorFull(input, output, shift, axes);
        }

        return ND4J_STATUS_OK;
    }
}
}

#endif