//
// Created by raver119 on 12.10.2017.
//
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {

        void _buildChunk(std::vector<int>& target, std::vector<int>& source, int rank, int offset) {
            for (int e = offset; e < offset + rank; e++)
                target.push_back(source[e]);
        }

        CUSTOM_OP_IMPL(strided_slice, 1, 1, true, 0, -1) {
            auto x = INPUT_VARIABLE(0);


            nd4j_debug("x rank: %i\n", x->rankOf());

            REQUIRE_TRUE(block.getIArguments()->size() == x->rankOf() * 3, 0, "Number of Integer arguments should be equal to input rank x 3 = %i, but got %i instead", (x->rankOf() * 3), block.getIArguments()->size());

            std::vector<int> begin;
            std::vector<int> end;
            std::vector<int> strides;

            _buildChunk(begin, *(block.getIArguments()), x->rankOf(), 0);
            _buildChunk(end, *(block.getIArguments()), x->rankOf(), x->rankOf());
            _buildChunk(strides, *(block.getIArguments()), x->rankOf(), x->rankOf() * 2);


            nd4j::Logger::printv("begin", begin);
            nd4j::Logger::printv("end", end);
            nd4j::Logger::printv("strides", strides);


            auto z = OUTPUT_VARIABLE(0);

            Nd4jIndex offset = 0;
            Nd4jIndex length = 1;
            std::vector<int> newShape;
            IndicesList indices;
            for (int e = 0; e < x->rankOf(); e++) {
                auto start = begin[e];
                auto stop = end[e];
                auto stride = strides[e];
                auto elements = (stop - start) / stride;

                nd4j_debug("Dimension [%i]; elements: [%i]\n", e, elements);

                indices.push_back(NDIndex::interval(start, stop, stride));
            }

            nd4j_debug("Final offset: %i; Final length: %i\n", (int) offset, (int) length);
            nd4j::Logger::printv("Final shape", newShape);

            auto sub = x->subarray(indices);
            z->assign(sub);

            STORE_RESULT(*z);
            delete sub;

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(stridedslice, strided_slice);

        DECLARE_SHAPE_FN(strided_slice) {
            auto inShape = inputShape->at(0);

            std::vector<int> begin;
            std::vector<int> end;
            std::vector<int> strides;

            int x_rank = shape::rank(inShape);

            _buildChunk(begin, *(block.getIArguments()), x_rank, 0);
            _buildChunk(end, *(block.getIArguments()), x_rank, x_rank);
            _buildChunk(strides, *(block.getIArguments()), x_rank, x_rank * 2);

            int *newShape;

            Nd4jIndex length = 1;
            std::vector<int> shape;
            for (int e = 0; e < shape::rank(inShape); e++) {
                auto start = begin[e];
                auto stop = end[e];
                auto stride = strides[e];

                int elements = 0;
                for (int i = start; i < stop; i += stride)
                    elements++;

                shape.push_back(elements);

                length *= elements;
            }

            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            shape::shapeBuffer(x_rank, shape.data(), newShape);

            return new ShapeList(newShape);
        }
    }
}