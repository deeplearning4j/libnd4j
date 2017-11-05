//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_DECLARABLE_LIST_OP_H
#define LIBND4J_DECLARABLE_LIST_OP_H

#include <graph/Block.h>
#include <ops/declarable/DeclarableOp.h>

using namespace nd4j::graph;

namespace nd4j {
    namespace ops {
        template <typename T>
        class DeclarableListOp : nd4j::ops::DeclarableOp<T> {
        protected:
            virtual Nd4jStatus validateAndExecute(Block<T>& block) = 0;

        public:
            DeclarableListOp(int numInputs, int numOutputs, const char* opName, int tArgs, int iArgs);
            ~DeclarableListOp();

            void execute(Block<T>& block);
        };
    }
}

#endif