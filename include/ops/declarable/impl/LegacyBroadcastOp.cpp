//
// Created by raver119 on 17.10.2017.
//

#include <ops/declarable/LegacyBroadcastOp.h>


namespace nd4j {
    namespace ops {
        template <typename T>
        Nd4jStatus LegacyBroadcastOp<T>::validateAndExecute(Block<T> &block) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);

            auto z = OUTPUT_VARIABLE(0);

            std::vector<int> dims(*block.getIArguments());
            if (dims.size() > 0)
                std::sort(dims.begin(), dims.end());


            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            shape::TAD tad(x->shapeInfo(), dims.data(), dims.size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();

            if (x == z)
                NativeOpExcutioner<T>::execBroadcast(opNum, x->buffer(), x->shapeInfo(), y->buffer(), y->shapeInfo(), z->buffer(), z->shapeInfo(), dims.data(), dims.size(), tad.tadOnlyShapeInfo, tad.tadOffsets, tad.tadOnlyShapeInfo, tad.tadOffsets);
            else {
                shape::TAD tadZ(z->shapeInfo(), dims.data(), dims.size());
                tadZ.createTadOnlyShapeInfo();
                tadZ.createOffsets();

                NativeOpExcutioner<T>::execBroadcast(opNum, x->buffer(), x->shapeInfo(), y->buffer(), y->shapeInfo(), z->buffer(), z->shapeInfo(), dims.data(), dims.size(), tad.tadOnlyShapeInfo, tad.tadOffsets, tadZ.tadOnlyShapeInfo, tadZ.tadOffsets);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        template <typename T>
        LegacyBroadcastOp<T>::LegacyBroadcastOp() : LegacyOp<T>::LegacyOp(2) {
            //
        }

        template <typename T>
        LegacyBroadcastOp<T>::LegacyBroadcastOp(int opNum) : LegacyOp<T>::LegacyOp(2, opNum) {
            //
        }

        template <typename T>
        ShapeList* LegacyBroadcastOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Block<T> &block) {
            auto inShape = inputShape->at(0);

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            memcpy(newShape, inShape, shape::shapeInfoByteLength(inShape));

            return new ShapeList(newShape);
        }


        template class LegacyBroadcastOp<float>;
        template class LegacyBroadcastOp<float16>;
        template class LegacyBroadcastOp<double>;
    }
}
