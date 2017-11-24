//
// Created by raver119 on 24.11.17.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(scatter_add, 3, 1, true) {
            auto input = INPUT_VARIABLE(0);
            auto indices = INPUT_VARIABLE(1);
            auto updates = INPUT_VARIABLE(2);

            auto output = OUTPUT_VARIABLE(0);

            if (!block.isInplace())
                output->assign(input);

            if (indices->isVector() && input->isVector() && updates->isVector()) {
                for (int e = 0; e < indices->lengthOf(); e++) {
                    int idx = (int) indices->getScalar(e);
                    
                    T t0 = input->getScalar(idx);
                    T t1 = updates->getScalar(e);
                    
                    output->putScalar(idx, t0 + t1);
                }
            } else if (indices->isVector() || (indices->isScalar() && indices->rankOf() == input->rankOf())) {
                std::vector<int> idc;
                std::vector<int> idcU;

                for (int e = 0; e < indices->lengthOf(); e++) {
                    idc.push_back((int) indices->getScalar(e));
                    idcU.push_back(e);
                }

                std::vector<int> tadDimension = ShapeUtils<T>::convertAxisToTadTarget(input->rankOf(), {0});
                auto tadsOperand = NDArrayFactory<T>::multipleTensorsAlongDimension(output, idc, tadDimension);
                auto tadsUpdate = NDArrayFactory<T>::multipleTensorsAlongDimension(updates, idcU, tadDimension);

                for (int e = 0; e < tadsOperand->size(); e++) {
                    auto t0 = tadsOperand->at(e);
                    auto t1 = tadsUpdate->at(e);

                    t0->template applyPairwiseTransform<simdOps::Add<T>>(t1, nullptr);
                }

                delete tadsOperand;
                delete tadsUpdate;
            }

            return ND4J_STATUS_OK;
        }
    }
}