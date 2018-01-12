//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/parity_ops.h>

namespace nd4j {
namespace ops {
    OP_IMPL(cross, 2, 1, false) {
        auto a = INPUT_VARIABLE(0);
        auto b = INPUT_VARIABLE(1);

        REQUIRE_TRUE(a->lengthOf() == b->lengthOf(), 0, "Cross: A and B lengths should match");
        REQUIRE_TRUE(a->rankOf() >= 1 && b->rankOf() >= 1, 0, "Cross: A and B should have rank >= 1");

        // TODO: we might want to lift this restriction
        REQUIRE_TRUE(a->isSameShape(b),0, "Cross: A and B should have equal shape");

        auto o = OUTPUT_VARIABLE(0);

        if (a->lengthOf() == 3) {
            auto a0 = a->getScalar(0);
            auto a1 = a->getScalar(1);
            auto a2 = a->getScalar(2);

            auto b0 = b->getScalar(0);
            auto b1 = b->getScalar(1);
            auto b2 = b->getScalar(2);

            o->putScalar(0, a1 * b2 - a2 * b1);
            o->putScalar(1, a2 * b0 - a0 * b2);
            o->putScalar(2, a0 * b1 - a1 * b0);
        } else {
            REQUIRE_TRUE(a->sizeAt(-1) == 3, 0, "Cross: outer dimension of A and B should be equal to 3");
            auto _a = a->reshape(a->ordering(), {-1, 3});
            auto _b = b->reshape(b->ordering(), {-1, 3});
            auto _o = o->reshape(o->ordering(), {-1, 3});

            auto tadsA = NDArrayFactory<T>::allTensorsAlongDimension(_a, {1});
            auto tadsB = NDArrayFactory<T>::allTensorsAlongDimension(_b, {1});
            auto tadsO = NDArrayFactory<T>::allTensorsAlongDimension(_o, {1});

            int tads = tadsA->size();

            #pragma omp parallel for simd schedule(static)
            for (int e = 0; e < tads; e++) {
                auto a_ = tadsA->at(e);
                auto b_ = tadsB->at(e);
                auto o_ = tadsO->at(e);

                auto a0 = a_->getScalar(0);
                auto a1 = a_->getScalar(1);
                auto a2 = a_->getScalar(2);

                auto b0 = b_->getScalar(0);
                auto b1 = b_->getScalar(1);
                auto b2 = b_->getScalar(2);

                o_->putScalar(0, a1 * b2 - a2 * b1);
                o_->putScalar(1, a2 * b0 - a0 * b2);
                o_->putScalar(2, a0 * b1 - a1 * b0);
            }

            delete tadsA;
            delete tadsB;
            delete tadsO;
            delete _a;
            delete _b;
            delete _o;
        }

        return ND4J_STATUS_OK;
    }
}
}