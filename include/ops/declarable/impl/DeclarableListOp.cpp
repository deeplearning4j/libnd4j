//
// @author raver119@gmail.com
//

#include <ops/declarable/OpDescriptor.h>
#include <ops/declarable/DeclarableListOp.h>

namespace nd4j {
    namespace ops {
        template <typename T>
        DeclarableListOp<T>::~DeclarableListOp() {
            //
        }

        template <typename T>
        DeclarableListOp<T>::DeclarableListOp(int numInputs, int numOutputs, const char* opName, int tArgs, int iArgs) : DeclarableOp<T>::DeclarableOp(numInputs, numOutputs, opName, false, tArgs, iArgs) {
            // This kind of operations work with sets: NDArrayList
            this->getOpDescriptor()->setInputType(InputType_NUMERIC_SET);
        }

        template <typename T>
        void DeclarableListOp<T>::execute(Block<T>& block)  {
            //
        }

        /**
         * This method just outputs scalar buffer
         *
         * @tparam T
         * @param inputShape
         * @param block
         * @return
         */
        template <typename T>
        ShapeList* DeclarableListOp<T>::calculateOutputShape(ShapeList* inputShape, nd4j::graph::Block<T>& block) {
            // TODO: ensure this method isn't ever called

            std::vector<int> shape({1, 1});
            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), int);
            shape::shapeBuffer(2, shape.data(), newShape);

            return new ShapeList(newShape);
        }
    }
}