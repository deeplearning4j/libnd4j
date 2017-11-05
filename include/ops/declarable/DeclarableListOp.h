//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_DECLARABLE_LIST_OP_H
#define LIBND4J_DECLARABLE_LIST_OP_H

#include <graph/Block.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/DeclarableOp.h>

using namespace nd4j::graph;

namespace nd4j {
    namespace ops {
        template <typename T>
        class DeclarableListOp : public nd4j::ops::DeclarableOp<T> {
        protected:
            virtual Nd4jStatus validateAndExecute(Block<T>& block) = 0;

        public:
            DeclarableListOp(int numInputs, int numOutputs, const char* opName, int tArgs, int iArgs);
            ~DeclarableListOp();

            /*
            void execute(Block<T>& block);
*/
            Nd4jStatus execute(NDArrayList<T>* list, std::initializer_list<NDArray<T>*> inputs, std::initializer_list<T> tArgs, std::initializer_list<int> iArgs);
            Nd4jStatus execute(NDArrayList<T>* list, std::vector<NDArray<T>*> inputs, std::vector<T> tArgs, std::vector<int> iArgs);

            ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Block<T>& block) override;
        };
    }
}

#endif