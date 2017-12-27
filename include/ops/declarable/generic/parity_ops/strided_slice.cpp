//
// Created by raver119 on 12.10.2017.
//
#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>
#include <helpers/BitwiseUtils.h>

namespace nd4j {
    namespace ops {

        void _preprocess_strided_slice(IndicesList* indicesList, std::vector<int>* final_shape, std::vector<int>& input_shape, std::vector<int>& begin, std::vector<int>& end, std::vector<int>& strides, int begin_mask, int ellipsis_mask, int end_mask, int new_axis_mask, int shrink_axis_mask) {
            std::vector<int> shrinks = BitwiseUtils::valueBits(shrink_axis_mask);

            for (int e = 0; e < (int) input_shape.size(); e++) {

            }
        }


        CUSTOM_OP_IMPL(strided_slice, 1, 1, false, 0, 5) {
            auto x = INPUT_VARIABLE(0);

            int begin_mask = INT_ARG(0);
            int ellipsis_mask = INT_ARG(1);
            int end_mask = INT_ARG(2);
            int new_axis_mask = INT_ARG(3);
            int shrink_axis_mask = INT_ARG(4);

            int dim_values = 0; //block.getIArguments()->size() - 5;
            int delta = 0; //dim_values % 3;
            int elements = 0; //dim_values / 3;

            std::vector<int> begin;
            std::vector<int> end;
            std::vector<int> strides;

            bool isLive = false;

            std::vector<int> args;

            // statically evaluated 
            if (block.getIArguments()->size() > 5) {
                dim_values = block.getIArguments()->size() - 5;
                delta = dim_values % 3;
                elements = dim_values / 3;

                for (int e = 5; e < block.getIArguments()->size(); e++)
                    args.emplace_back(INT_ARG(e));

            } else if (block.width() >= 3) {
                isLive = true;

                auto v_begin = INPUT_VARIABLE(1);
                auto v_end = INPUT_VARIABLE(2);

                elements = v_begin->lengthOf();

                REQUIRE_TRUE(v_begin->lengthOf() == v_end->lengthOf(), 0, "Length of begin/end should match, but got %i vs %i instead", (int) v_begin->lengthOf(), (int) v_end->lengthOf());

                for (int e = 0; e < v_begin->lengthOf(); e++)
                    args.emplace_back((int) v_begin->getIndexedScalar(e));

                for (int e = 0; e < v_end->lengthOf(); e++)
                    args.emplace_back((int) v_end->getIndexedScalar(e));

                if (block.width() >= 4) {
                    auto v_stride = INPUT_VARIABLE(3);

                    REQUIRE_TRUE(v_stride->lengthOf() == v_begin->lengthOf(), 0, "Length of begin/end/stride should match, but got %i vs %i vs %i instead", (int) v_begin->lengthOf(), (int) v_end->lengthOf(), (int) v_stride->lengthOf());

                    for (int e = 0; e < v_stride->lengthOf(); e++)
                        args.emplace_back((int) v_stride->getIndexedScalar(e));
                } else {
                    for (int e = 0; e < v_begin->lengthOf(); e++)
                        args.emplace_back(1);
                }
            } else {
                REQUIRE_TRUE(false, 0, "Can't find begin/end/stride information neither in IArguments or in input arrays");
            }

            REQUIRE_TRUE(delta == 0, 0, "Number of Integer arguments should be equal to input rank x 3 = %i, but got %i instead", (x->rankOf() * 3), dim_values);

            int ellipsis = -1;
            if (ellipsis_mask != 0)
                ellipsis = BitwiseUtils::valueBit(ellipsis_mask);

            ShapeUtils<T>::copyVectorPart(begin, args, elements, 0);
            ShapeUtils<T>::copyVectorPart(end, args, elements, elements);
            ShapeUtils<T>::copyVectorPart(strides, args, elements, elements * 2);

            auto z = OUTPUT_VARIABLE(0);

            IndicesList indices;
            std::vector<int> input_shape = x->getShapeAsVector();
            _preprocess_strided_slice(&indices, nullptr, input_shape, begin, end, strides, begin_mask, ellipsis_mask, end_mask, new_axis_mask, shrink_axis_mask);

            auto sub = x->subarray(indices);

            if (!isLive)
                z->assign(sub);
            else {
                auto stb = sub->dup(x->ordering());
                OVERWRITE_RESULT(stb);
            }

            delete sub;

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(stridedslice, strided_slice);

        DECLARE_SHAPE_FN(strided_slice) {
            auto inShape = inputShape->at(0);

            int begin_mask = INT_ARG(0);
            int ellipsis_mask = INT_ARG(1);
            int end_mask = INT_ARG(2);
            int new_axis_mask = INT_ARG(3);
            int shrink_axis_mask = INT_ARG(4);

            int x_rank = shape::rank(inShape);

            int dim_values = block.getIArguments()->size() - 5;
            int delta = dim_values % 3;
            int elements = dim_values / 3;

            int delta2 = dim_values / x_rank;

            std::vector<int> begin;
            std::vector<int> end;
            std::vector<int> strides;

            std::vector<int> args;
            for (int e = 5; e < block.getIArguments()->size(); e++)
                args.emplace_back(INT_ARG(e));

            ShapeUtils<T>::copyVectorPart(begin, args, elements, 0);
            ShapeUtils<T>::copyVectorPart(end, args, elements, elements);
            ShapeUtils<T>::copyVectorPart(strides, args, elements, elements * 2);

            int *newShape;
            std::vector<int> input_shape(shape::rank(inShape));
            std::vector<int> shape;

            for (int e = 0; e < shape::rank(inShape); e++)
                input_shape[e] = shape::shapeOf(inShape)[e];

            _preprocess_strided_slice(nullptr, &shape, input_shape, begin, end, strides, begin_mask, ellipsis_mask, end_mask, new_axis_mask, shrink_axis_mask);

            nd4j_printv("shape after shrink: ", shape);

            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(shape.size()), int);
            shape::shapeBuffer(shape.size(), shape.data(), newShape);

            return new ShapeList(newShape);
        }
    }
}