//
// @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(unstack, 1, -1, false, 0, 1) {
            auto input = INPUT_VARIABLE(0);

            auto dim = INT_ARG(0);
            if (dim < 0)
                dim += input->rankOf();


            REQUIRE_TRUE(dim < input->rankOf(), 0, "Unstack dimension should be lower then rank of input");
            REQUIRE_TRUE(dim >= 0, 0, "Unstack dimension should be non-negative value");

            std::vector<int> dims;
            for (int e = 0; e < input->rankOf(); e++)
                if (e != dim)
                    dims.emplace_back(e);

            auto tads = NDArrayFactory<T>::allTensorsAlongDimension(input, dims);
            //nd4j_printf("Tad size: %d\n",tads->size());
            for (int e = 0; e < tads->size(); e++) {
                //nd4j_printf("Calling assign at index %d\n",e);
                auto outE = OUTPUT_VARIABLE(e);
                auto tadAtE = tads->at(e);

                outE->assign(tadAtE);

                this->storeResult(block, e, *outE);
            }

            delete tads;

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(unpack, unstack);
        
        DECLARE_SHAPE_FN(unstack) {
            auto inShape = inputShape->at(0);

            auto dim = INT_ARG(0);
            if (dim < 0)
                dim += shape::rank(inShape);

            std::vector<int> dims;
            for (int e = 0; e < shape::rank(inShape); e++)
                if (e != dim)
                    dims.emplace_back(e);

            shape::TAD tad(inShape, dims.data(), (int) dims.size());
            tad.createTadOnlyShapeInfo();
            Nd4jIndex numTads = shape::length(inShape) / shape::tadLength(inShape, dims.data(), (int) dims.size());
            
            std::vector<int> shape(shape::rank(tad.tadOnlyShapeInfo));
            for (int e = 0; e < shape::rank(tad.tadOnlyShapeInfo); e++)
                shape[e] = shape::shapeOf(tad.tadOnlyShapeInfo)[e];

            // remove leading 1
            if (shape.size() == 2 && shape[0] == 1)
                shape.erase(shape.begin());
            
            auto result = SHAPELIST();
            for (int e = 0; e < numTads; e++) {
                int *newShape;
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(shape.size()), int);
                if (shape::order(inShape) == 'c')
                    shape::shapeBuffer(shape.size(), shape.data(), newShape);
                else
                    shape::shapeBufferFortran(shape.size(), shape.data(), newShape);
                result->push_back(newShape);
            }

            return result;
        }
    }
}