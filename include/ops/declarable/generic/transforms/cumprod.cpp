//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/prefix.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(cumprod, 1, 1, true, 0, 2) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            const bool exclusive = INT_ARG(0) == 1;
            const bool reverse = INT_ARG(1) == 1;

            if (block.getIArguments()->size() == 0 && block.width() == 1) {
                // all at once case
                nd4j::ops::helpers::_prefix<T, simdOps::Multiply<T>>(input->buffer(), input->shapeInfo(), output->buffer(), output->shapeInfo(), exclusive, reverse);
            } else {
                std::vector<int> dims =  *(block.getIArguments());

                nd4j::ops::helpers::_prefix<T, simdOps::Multiply<T>>(input, output, dims, exclusive, reverse);
            }

            return ND4J_STATUS_OK;
        }
    }
}