//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/convo.h>
#include <ops/declarable/helpers/dilation2d.h>
#include <helpers/ShapeBuilder.h>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(dilation2d, 2, 1, false, 0, 1) {
        auto input = INPUT_VARIABLE(0);
        auto weights = INPUT_VARIABLE(1);

        auto output = OUTPUT_VARIABLE(0);

        return Status::OK();
    }

    DECLARE_SHAPE_FN(dilation2d) {
        auto input = inputShape->at(0);
        auto weights = inputShape->at(1);

        const int batch_size = shape::sizeAt(input, 0);
        const int depth = shape::sizeAt(input, 3);
        const bool isSameShape = INT_ARG(0) == 1;

        std::vector<int> strides(4);
        std::vector<int> rates(4);

        int *newShape;

        if (block.width() > 2) {

        } else {
            if (block.numI() < 9) {
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(0), int);
                ShapeBuilder::shapeScalar(newShape);
                return new ShapeList(newShape);
            }
                
            int e = 1;
            for (int cnt = 0; cnt < 4; cnt++)
                strides[cnt] = INT_ARG(e++);

            for (int cnt = 0;cnt < 4; cnt++)
                rates[cnt] = INT_ARG(e++);
        }

        int stride_rows = 0, stride_cols = 0;
        int rate_rows = 0, rate_cols = 0;
        int pad_top = 0, pad_left = 0;
        int out_rows = 0, out_cols = 0;

        helpers::_dilation_hw(input, weights, strides, rates, isSameShape, &stride_rows, &stride_cols, &rate_rows, &rate_cols, &pad_top, &pad_left, &out_rows, &out_cols);

        std::array<int, 4> shape = {{batch_size, out_rows, out_cols, depth}};
        ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(4), int);
        shape::shapeBuffer(4, shape.data(), newShape);

        shape::printShapeInfoLinear(newShape);

        return new ShapeList(newShape);
    }
}
}