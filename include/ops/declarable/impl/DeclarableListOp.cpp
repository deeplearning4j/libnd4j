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
            this->_descriptor->_inputType = InputType_NUMERIC_SET;
        }

        template <typename T>
        void DeclarableListOp<T>::execute(Block<T>& block) override {

        }
    }
}