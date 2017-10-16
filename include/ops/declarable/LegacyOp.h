//
// Created by raver119 on 16.10.2017.
//

#ifndef LIBND4J_LEGACYOP_H
#define LIBND4J_LEGACYOP_H

#include <ops/declarable/DeclarableOp.h>

namespace nd4j {
    namespace ops {
        template <typename T>
        class LegacyOp : public DeclarableOp<T> {
        protected:
            // this field is mainly for debugging
            int _opNum = -1;

            virtual Nd4jStatus validateAndExecute(Block<T>& block) = 0;
        public:
            LegacyOp(int numInputs);
            LegacyOp(int numInputs, int opNum);

            virtual ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Block<T>& block) = 0;
        };
    }
}


#endif //LIBND4J_LEGACYOP_H
