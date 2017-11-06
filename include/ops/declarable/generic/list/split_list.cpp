//
// Created by raver119 on 06.11.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(split_list, 2, 1, 0, -1) {
            NDArrayList<T> *list = nullptr;
            NDArray<T>* array = nullptr;
            NDArray<T>* sizes = nullptr;

            bool hasList = false;

            if (block.width() == 3){
                list = INPUT_LIST(0);
                array = INPUT_VARIABLE(1);
                sizes = INPUT_VARIABLE(2);
                hasList = true;
            } else {
                array = INPUT_VARIABLE(0);
                sizes = INPUT_VARIABLE(1);
                list = new NDArrayList<T>(sizes->lengthOf(), false);
            }

            // now let's build subarrays
            int cnt = 0;
            for (int e = 0; e < sizes->lengthOf(); e++) {
                int c_size = (int) sizes->getIndexedScalar(e);
                IndicesList indices;

                // we're adding our interval along zeroth dimension
                indices.push_back(NDIndex::interval(cnt, cnt+c_size));

                // and then we set all other dimensions to All
                for (int e = 1; e < array->rankOf(); e++)
                    indices.push_back(NDIndex::all());


                auto subarray = array->subarray(indices);

                auto status = list->write(e, subarray->dup(array->ordering()));
                if (status != ND4J_STATUS_OK)
                    return status;

                delete subarray;

                cnt += c_size;
            }

            if (!hasList) {
                OVERWRITE_RESULT(list);
            }

            return ND4J_STATUS_OK;
        }
    }
}