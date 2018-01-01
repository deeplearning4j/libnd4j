//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/parity_ops.h>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(split, 2, -1, false, 0, -2) {
        auto input = INPUT_VARIABLE(0);
        auto sizes = INPUT_VARIABLE(1);

        int axis = 0;

        if (block.width() > 2){
            auto _a = INPUT_VARIABLE(2);
            axis = _a->getScalar(0);
        } else if (block.getIArguments()->size() > 0) {
            axis = INT_ARG(0);
        };

        std::vector<int> dims = ShapeUtils<T>::convertAxisToTadTarget(input->rankOf(), {axis});
        auto tads = NDArrayFactory<T>::allTensorsAlongDimension(input, dims);

        for (int e = 0; e < sizes->lengthOf(); e++) {
            
        }

        delete tads;
        return ND4J_STATUS_OK;
    }


    DECLARE_SHAPE_FN(split) {
        return new ShapeList();
    }
}
}