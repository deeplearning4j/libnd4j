//
// Created by raver119 on 16.10.2017.
//

#include <ops/declarable/LegacyRandomOp.h>

#include <NativeOpExcutioner.h>


namespace nd4j {
    namespace ops {
        template <typename T>
        LegacyRandomOp<T>::LegacyRandomOp() : LegacyOp<T>::LegacyOp(1) {
            // just a no-op
        }

        template <typename T>
        LegacyRandomOp<T>::LegacyRandomOp(int opNum) : LegacyOp<T>::LegacyOp(1, opNum) {
            // just a no-op
        }

        template <typename T>
        Nd4jStatus LegacyRandomOp<T>::validateAndExecute(Context<T> &block) {
            auto input = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            //NativeOpExcutioner<T>::execTransform(opNum, input->getBuffer(), input->getShapeInfo(), z->getBuffer(), z->getShapeInfo(), block.getTArguments()->data(), nullptr, nullptr);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        /**
        * For transform operations, output shape always equals to input shape. With just a few exclusions, like im2col and col2im. 
        * But these ops already have CustomOp implementations.
        *
        */
        template <typename T>
        ShapeList *LegacyRandomOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context<T> &ctx) {
            auto inShape = inputShape->at(0);

            int *newShape;
            ALLOCATE(newShape, ctx.getWorkspace(), shape::shapeInfoLength(inShape), int);
            memcpy(newShape, inShape, shape::shapeInfoByteLength(inShape));

            return new ShapeList(newShape);
        }


        template class ND4J_EXPORT LegacyRandomOp<float>;
        template class ND4J_EXPORT LegacyRandomOp<double>;
        template class ND4J_EXPORT LegacyRandomOp<float16>;
    }
}