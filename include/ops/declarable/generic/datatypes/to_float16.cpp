//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation casts elements of input array to float16 data type
         * 
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         */
        OP_IMPL(to_float16, 1, 1, true) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            if (!block.isInplace())
                output->assign(input);

            STORE_RESULT(output);

            return ND4J_STATUS_OK;
        }
    }
}