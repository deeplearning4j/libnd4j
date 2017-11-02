//
// Created by raver119 on 02.11.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(slice, 1, 1, false, 0, -1) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            std::vector<int> begin;
            std::vector<int> end;

            int x_rank = input->rankOf();

            ShapeUtils<T>::copyVectorPart(begin, *(block.getIArguments()), x_rank, 0);
            ShapeUtils<T>::copyVectorPart(end, *(block.getIArguments()), x_rank, x_rank);

            IndicesList indices;
            for (int e = 0; e < x_rank; e++) {
                int size = end[e] - begin[e];
                REQUIRE_TRUE(size > 1, 0, "Slice interval for dimension %i is less then 1", e);

                indices.push_back(NDIndex::interval(begin[e], end[e]));
            }
            auto sub = input->subarray(indices);
            output->assign(sub);

            delete sub;

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(slice) {
            auto inShape = inputShape->at(0);

            std::vector<int> begin;
            std::vector<int> end;

            int x_rank = shape::rank(inShape);

            ShapeUtils<T>::copyVectorPart(begin, *(block.getIArguments()), x_rank, 0);
            ShapeUtils<T>::copyVectorPart(end, *(block.getIArguments()), x_rank, x_rank);

            int *newShape;
            std::vector<int> shape;
            for (int e = 0; e < x_rank; e++) {
                int size = end[e] - begin[e];
                shape.push_back(size);
            }

            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(x_rank), int);
            shape::shapeBuffer(x_rank, shape.data(), newShape);

            return new ShapeList(newShape);
        }
    }
}