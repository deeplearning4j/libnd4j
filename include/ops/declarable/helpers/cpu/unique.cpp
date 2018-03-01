//
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/unique.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int uniqueCount(NDArray<T>* input) {
        int count = 0;

        std::vector<T> valuesVector;

        for (int e = 0; e < x->lengthOf(); e++) {
            T v = x->getScalar(e);
            if (std::find(valuesVector.begin(), values.end(), v) == values.end()) {
                valuesVector.push_back(v);
                count++;
            }
        }
        return count;
    }

    template int uniqueCount(NDArray<float>* input);
    template int uniqueCount(NDArray<float16>* input);
    template int uniqueCount(NDArray<double>* input);


    template <typename T>
    int uniqueFunctor(NDArray<T>* input, NDArray<T>* values, NDArray<T>* indices, NDArray<T>* counts) { 
    
        std::vector<T> valuesVector;
        std::map<T, int> indicesMap;
        std::map<T, int> countsMap;

        for (int e = 0; e < x->lengthOf(); e++) {
            T v = x->getScalar(e);
            if (std::find(valuesVector.begin(), values.end(), v) == values.end()) {
                valuesVector.push_back(v);
                indicesMap[v] = e;
                countsMap[v] = 1;
            }
            else {
                countsMap[v]++;
            }
        }


        for (int e = 0; e < values->lengthOf(); e++) {
            values->putScalar(e, valuesVector[e]);
        }

        for (int e = 0; e < indices->lengthOf(); e++) {
            indices->putScalar(e, indicesMap[(*input)(e)]);
        }
        
        if (counts != nullptr) {
            for (int e = 0; e < counts->lengthOf(); e++) {
                values->putScalar(e, countsMap[valuesVector[e]]);
            }
        }
        return ND4J_STATUS_OK;
    }

    template int uniqueFunctor(NDArray<float>* input, NDArray<float>* values, NDArray<float>* indices, NDArray<float>* counts);
    template int uniqueFunctor(NDArray<float16>* input, NDArray<float16>* values, NDArray<float16>* indices, NDArray<float16>* counts);
    template int uniqueFunctor(NDArray<double>* input, NDArray<double>* values, NDArray<double>* indices, NDArray<double>* counts);

}
}
}