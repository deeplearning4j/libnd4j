#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(confusion_matrix, 2, 1, false, 0, -2) {

            auto labels = INPUT_VARIABLE(0);
            auto predictions = INPUT_VARIABLE(1);


            REQUIRE_TRUE(labels->isVector(), 0, "Labels input should be Vector, but got %iD instead", input->rankOf());
            REQUIRE_TRUE(predictions->isVector(), 0, "Predictions input should be Vector, but got %iD instead", input->rankOf());
            REQUIRE_TRUE(labels>isSameShape(predictions),0, "Labels and predictions should have equal shape");

            auto output = OUTPUT_VARIABLE(0);



            // code below is wrong
            NDArray<T>* output = OUTPUT_VARIABLE(0);

            const int rank = output->rankOf();
            ResultSet<T>* arrs = NDArrayFactory<T>::allTensorsAlongDimension(output, {rank-2, rank-1});

            for(int i = 0; i < arrs->size(); ++i)
                arrs->at(i)->setIdentity();

            delete arrs;


            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(confusion_matrix) {
            int *inShape = inputShape->at(0);


            int numClasses = 0;

            if (block.getIArguments()->size() > 0) {
                numClasses = INT_ARG(0);
            }
            else  {
                // the one plus the maximum value in either predictions or labels array
                // numClasses =  ?

            }


            int* outShapeInfo(nullptr);

            ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(2), int);
            outShapeInfo[0] = 2;
            outShapeInfo[1] = numClasses;
            outShapeInfo[2] = numClasses;

            return new ShapeList(newShape);
        }
    }
}
