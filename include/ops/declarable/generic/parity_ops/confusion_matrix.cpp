#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>
#include <NDArray.h>
#include <array/NDArrayList.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(confusion_matrix, 2, 1, false, 0, -2) {

            auto labels = INPUT_VARIABLE(0);
            auto predictions = INPUT_VARIABLE(1);
            NDArray<T>* weights = nullptr;
            if(block.width() > 2){
                weights = INPUT_VARIABLE(2);
                REQUIRE_TRUE(weights->isSameShape(predictions),0, "Weights and predictions should have equal shape");
            }
            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(labels->isVector(), 0, "Labels input should be a Vector, but got %iD instead", labels->rankOf());
            REQUIRE_TRUE(predictions->isVector(), 0, "Predictions input should be Vector, but got %iD instead", predictions->rankOf());
            REQUIRE_TRUE(labels->isSameShape(predictions),0, "Labels and predictions should have equal shape");

            ResultSet<T>* arrs = NDArrayFactory<T>::allTensorsAlongDimension(output, {1});


            for (int j = 0; j < labels->lengthOf(); ++j){
                Nd4jIndex label = labels->getScalar(j);
                Nd4jIndex pred = predictions->getScalar(j);
                T value = (weights==nullptr) ? (T)1.0 : weights->getScalar(j);
                arrs->at(label)->putScalar(pred, value);
            }

            delete arrs;

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(confusion_matrix) {

            auto labels = INPUT_VARIABLE(0);
            auto predictions = INPUT_VARIABLE(1);

            int numClasses = 0;

            int minPrediction = predictions->template reduceNumber<simdOps::Min<T>>();

            if(minPrediction <0)
                throw "Predictions contains negative values";

            int minLabel = labels->template reduceNumber<simdOps::Min<T>>();

            if(minLabel <0)
                throw "Labels contains negative values";

            if (block.getIArguments()->size() > 0) {
                numClasses = INT_ARG(0);
            }
            else  {

                int maxPrediction = predictions->template reduceNumber<simdOps::Max<T>>();
                int maxLabel = labels->template reduceNumber<simdOps::Max<T>>();
                if(maxPrediction >= maxLabel)
                    numClasses =  maxPrediction+1;
                else
                    numClasses = maxLabel+1;

            }

            int *newShape;
            std::array<int, 2> shape = {{numClasses,numClasses}};
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), int);
            shape::shapeBuffer(2, shape.data(), newShape);

            return new ShapeList(newShape);
        }
    }
}
