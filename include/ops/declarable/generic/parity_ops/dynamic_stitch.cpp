//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>
//#include <array>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(dynamic_stitch, 2, 1, false, 0, 0) {
        NDArray<T>* indices = INPUT_VARIABLE(0);
        NDArray<T>* firstData = INPUT_VARIABLE(1);
        int numOfData = block.width() - 1;
        NDArray<T>* output = OUTPUT_VARIABLE(0); 
        NDArray<T>* data = INPUT_VARIABLE(1);
        int k = 0;

        for (int e = 0, i = 1; e < indices->lengthOf(); e++) {
            T val = (*data)(k++); 
            nd4j_printf("%i. from %i\n", e, output->lengthOf());
            int pos = (*indices)(e);
            nd4j_printf("%i. pos is %i\n", e, pos);
            output->putScalar(pos, val);
            if (k >= data->lengthOf())
            {
                if (++i <= numOfData)
                    data = INPUT_VARIABLE(i);
                else
                    break;
                k = 0;
            }
        }
        
        return ND4J_STATUS_OK;
    }

    DECLARE_SHAPE_FN(dynamic_stitch) {
        int numOfData = block.width() - 1;
        NDArray<T>* input = INPUT_VARIABLE(0);
        int maxValue = input->getScalar(0);

        for (int e = 1; e < input->lengthOf(); ++e) {
            if (T(maxValue) < (*input)(e))
                maxValue = static_cast<int>((*input)(e));
        }

        auto shapes = new ShapeList();
        int *newShape;
        ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(1), int);

        ShapeBuilder::shapeVector(maxValue + 1, newShape);

        shapes->push_back(newShape);
        return shapes;
    }
}
}