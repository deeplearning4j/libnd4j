//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/parity_ops.h>

// this op will probably never become GPU-compatible
namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(listdiff, 2, 2, false, 0, 0) {
            auto values = INPUT_VARIABLE(0);
            auto keep = INPUT_VARIABLE(1);

            std::vector<T> saved;
            std::vector<T> indices;

            REQUIRE_TRUE(values->rankOf() == 1, 0, "ListDiff: rank of values should be 1D, but got %iD instead", values->rankOf());
            REQUIRE_TRUE(keep->rankOf() == 1, 0, "ListDiff: rank of keep should be 1D, but got %iD instead", keep->rankOf());

            for (int e = 0; e < values->lengthOf(); e++) {
                T v = values->getScalar(e);
                T extras[] = {v, (T) 0.0f, (T) 10.0f};
                auto idx = keep->template indexReduceNumber<simdOps::FirstIndex<T>>(extras);
                if (idx < 0) {
                    saved.emplace_back(v);
                    indices.emplace_back(e);
                }
            }

            // FIXME: we need 0-size NDArrays
            if (saved.size() == 0) {
                REQUIRE_TRUE(false, 0, "ListDiff: search returned no results");
            } else {
                auto z0 = OUTPUT_VARIABLE(0); //new NDArray<T>('c', {(int) saved.size()});
                auto z1 = OUTPUT_VARIABLE(1); //new NDArray<T>('c', {(int) saved.size()});

                memcpy(z0->buffer(), saved.data(), saved.size() * sizeof(T));
                memcpy(z1->buffer(), indices.data(), indices.size() * sizeof(T));

                OVERWRITE_2_RESULTS(z0, z1);
            }

            return ND4J_STATUS_OK;
        };

        DECLARE_SHAPE_FN(listdiff) {
            auto values = INPUT_VARIABLE(0);
            auto keep = INPUT_VARIABLE(1);

            int saved = 0;

            REQUIRE_TRUE(values->rankOf() == 1, 0, "ListDiff: rank of values should be 1D, but got %iD instead", values->rankOf());
            REQUIRE_TRUE(keep->rankOf() == 1, 0, "ListDiff: rank of keep should be 1D, but got %iD instead", keep->rankOf());

            for (int e = 0; e < values->lengthOf(); e++) {
                T v = values->getScalar(e);
                T extras[] = {v, (T) 0.0f, (T) 10.0f};
                auto idx = keep->template indexReduceNumber<simdOps::FirstIndex<T>>(extras);
                if (idx < 0)
                    saved++;
            }

            int *shapeX = ShapeUtils<T>::createVectorShapeInfo(saved, block.workspace());
            int *shapeY = ShapeUtils<T>::createVectorShapeInfo(saved, block.workspace());

            return SHAPELIST(shapeX, shapeY);
        }
    }
}