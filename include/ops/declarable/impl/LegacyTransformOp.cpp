//
// Created by raver119 on 16.10.2017.
//

#include "ops/declarable/LegacyTransformOp.h"

#include <NativeOpExcutioner.h>


namespace nd4j {
    namespace ops {
        template <typename T>
        LegacyTransformOp<T>::LegacyTransformOp() : LegacyOp<T>::LegacyOp(1) {
            // just a no-op
        }

        template <typename T>
        LegacyTransformOp<T>::LegacyTransformOp(int opNum) : LegacyOp<T>::LegacyOp(1, opNum) {
            // just a no-op
        }

        template <typename T>
        Nd4jStatus LegacyTransformOp<T>::validateAndExecute(Block<T> &block) {
            auto input = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            NativeOpExcutioner<T>::execTransform(opNum, input->getBuffer(), input->getShapeInfo(), z->getBuffer(), z->getShapeInfo(), block.getTArguments()->data(), nullptr, nullptr);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        template <typename T>
        ShapeList *LegacyTransformOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Block<T> &block) {
            auto inShape = inputShape->at(0);

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            memcpy(newShape, inShape, shape::shapeInfoByteLength(inShape));

            return new ShapeList(newShape);
        }


        template class LegacyTransformOp<float>;
        template class LegacyTransformOp<double>;
        template class LegacyTransformOp<float16>;
    }
}