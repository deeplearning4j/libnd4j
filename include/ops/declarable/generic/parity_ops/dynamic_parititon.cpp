//
//  @author @shugeo
//

#include <ops/declarable/CustomOperations.h>
#include <array>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(dynamic_partition, 2, 1, false, 0, 1) {
        NDArray<T>* input = INPUT_VARIABLE(0);
        NDArray<T>* indices = INPUT_VARIABLE(1);
        //NDArray<T>* indexVector = ; // linearize to actualize
        int numPartition = INT_ARG(0);
        std::vector<std::pair<NDArray<T>*, int>> outputs(numPartition);
        for (int i = 0; i < numPartition; i++)
        {
            outputs[i].first = OUTPUT_VARIABLE(i);
            outputs[i].second = 0;
            for (int e = 0; e < indices->lengthOf(); ++e)
                if ((*indices)(e) == T(i))
                    outputs[i].first->putScalar(outputs[i].second++, (*input)(e));
        }
        
        return ND4J_STATUS_OK;
    }

    DECLARE_SHAPE_FN(dynamic_partition) {
        int numPartition = INT_ARG(0);
        NDArray<T>* indices = INPUT_VARIABLE(1);
        std::vector<int> partitionSizes(numPartition, 0);
        int* in = inputShape->at(0);
        int* idx = inputShape->at(1); 
        for (int i = 0; i < numPartition; i++) {
            for (int e = 0; e < indices->lengthOf(); ++e)
                if ((*indices)(e) == T(i))
                    partitionSizes[i]++;
        }

        auto shapes = SHAPELIST();
        int outRank = shape::rank(in) - shape::rank(idx) + 1;
        for (int e = 0; e < numPartition; e++) {
            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(outRank), int);
            //shape::shapeVector(partitionSizes[e], newShape);
            newShape[0] = outRank;
            newShape[1] = partitionSizes[e];
            for(int i = 1; i < outRank; ++i)
                newShape[i + 1] = shape::sizeAt(in, outRank + i - 1);

            shape::updateStrides(newShape, shape::order(in));

            shapes->push_back(newShape);
        }

        return shapes;
    }
}
}