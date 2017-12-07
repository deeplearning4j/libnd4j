//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        REDUCTION_OP_IMPL(norm, 1, 1, false, 1, -2) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            int mode = (int) T_ARG(0);

            switch(mode) {
                case 0: {
                    // fro
                }
                break;
                case 1: {
                    // eunclidean
                }
                break;
                case 2: {
                    // 1
                    input->template reduceAlongDimension<simdOps::Norm1<T>>(output, *block.getIArguments());
                }
                break;
                case 3: {
                    // 2 
                    input->template reduceAlongDimension<simdOps::Norm2<T>>(output, *block.getIArguments());
                }
                break;
                case 4: {
                    // inf-norm
                    input->template reduceAlongDimension<simdOps::AMax<T>>(output, *block.getIArguments());
                }
                break;
                default: {
                    // p-norm
                }
            }

            return ND4J_STATUS_OK;
        };
    }
}