//
// Created by raver119 on 15.10.2017.
//

#ifndef LIBND4J_LOGICOP_H
#define LIBND4J_LOGICOP_H

#include "DeclarableOp.h"

namespace nd4j {
    namespace ops {

        /**
         * Logic ops are basically a placeholders. They do not have real viable implementation as ops.
         *
         * Their code is the part of GraphExecutioner logic. But we still want them to be expressed via Graph
         * @tparam T
         */
        template <typename T>
        class LogicOp : public DeclarableOp<T> {
        protected:
            Nd4jStatus validateAndExecute(nd4j::graph::Block<T>& block) override;
        public:
            LogicOp();
            ~LogicOp() = default;

            ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Block<T>& block) override;
        };
    }
}


#endif //LIBND4J_LOGICOP_H
