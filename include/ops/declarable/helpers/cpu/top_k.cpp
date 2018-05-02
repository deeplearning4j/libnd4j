//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/top_k.h>
#include <NDArrayFactory.h>
#include <ops/declarable/headers/parity_ops.h>
namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int topKFunctor(NDArray<T>* input, NDArray<T>* values, NDArray<T>* indeces, int k, bool needSort) {
        int width = input->sizeAt(-1);
        std::unique_ptr<ResultSet<T>> lastDimList(NDArrayFactory<T>::allTensorsAlongDimension(input, {input->rankOf() - 1}));
            if (k == 1) {
                int pos = 0;
#pragma omp parallel for if(lastDimList->size() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                for (int e = 0; e < lastDimList->size(); ++e) {
                    int maxPos = lastDimList->at(e)->argMax();
                    (*indeces)(e) = maxPos; //topIndex;
                    (*values)(e) = (*lastDimList->at(e))(maxPos);
                }
            }
            else { // if (k > 1) {

//                int width = input->sizeAt(-1);
                int nextPos = 0;
//#pragma omp parallel for 
                for (int e = 0; e < input->lengthOf(); e += width)
                {
                    std::vector<int> topIndeces(k);
                    std::vector<T>   topValues(k);
                    for (int pos = 0; pos < k; ++pos) {
                        topIndeces[pos] = pos;
                        topValues[pos] = (*input)(pos + e);
                    }
                    std::vector<T> sortedVals(topValues);
                    std::sort(sortedVals.begin(), sortedVals.end()); // sorted in ascending order

//#pragma omp parallel for 
                    for (int j = k; j < width; j++) {
                        T val = (*input)(j + e);
                        if (sortedVals[0] < val) { // value can be inserted to top k
                            if (sortedVals.end() == std::find(sortedVals.begin(), sortedVals.end(), val)) {    
                                // exchangePos - a distance between begin and minimum to be suppressed by val
                                auto exchangePos = std::distance(topValues.begin(), std::find(topValues.begin(), topValues.end(), sortedVals[0]));
//                                if ((exchangePos < 0 || exchangePos >= k), 1, "top_k: invalid index")
//                                    return ; 
                                // set up sorted sequence for continue
                                topValues[exchangePos] = val;
                                sortedVals[0] = val;
                                std::sort(sortedVals.begin(), sortedVals.end()); // sorted in ascending order
                            }
                        }
                    }

                    if (needSort) {
                        std::sort(topValues.begin(), topValues.end(), [](int a, int b) {
                            return a > b;   
                        });
                    }

//#pragma omp parallel for 
                    for (int j = 0; j < width; j++)
                        for (int pos = 0; pos < k; ++pos)
                            if (topValues[pos] == (*input)(j + e))
                                topIndeces[pos] = j;

                    for (int pos = 0; pos < k; ++pos, ++nextPos)
                    {
                        if (values != nullptr)
                            (*values)(nextPos)  =  topValues[pos];

                        (*indeces)(nextPos) = topIndeces[pos];
                    }
                }
        }
        return ND4J_STATUS_OK;
    }
// ----------------------------------------------------------------------------------------------- //

    template <typename T>
    int inTopKFunctor(NDArray<T>* input, NDArray<T>* target, NDArray<T>* result, int k) {

            std::vector<int> shapeV(input->rankOf() + 1);
            for (int i = 0; i < input->rankOf(); i++)
                shapeV[i] = input->sizeAt(i);
            shapeV[input->rankOf()] = k;
            std::unique_ptr<NDArray<T>> indices( new NDArray<T>(input->ordering(), shapeV));
            NDArray<T>* values = nullptr;
            int status = topKFunctor(input, values, indices.get(), k, true);

            if (status == ND4J_STATUS_OK) {
#pragma omp parallel for if(target->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                for (int e = 0; e < target->lengthOf(); e++) {
                    bool found = false;
                    for (int j = 0; j < k; j++) {
                        if ((*target)(e) == (*indices)(e * k + j)) {
                            found = true;
                            break;
                        }
                    }
                    if (found)
                        (*result)(e) = (T)1.f;
                }
            }
            return status; 

    }
    template int topKFunctor<float>(NDArray<float>* input, NDArray<float>* values, NDArray<float>* indeces, int k, bool needSort);
    template int topKFunctor<float16>(NDArray<float16>* input, NDArray<float16>* values, NDArray<float16>* indeces, int k, bool needSort);
    template int topKFunctor<double>(NDArray<double>* input, NDArray<double>* values, NDArray<double>* indeces, int k, bool needSort);
    template int inTopKFunctor<float>(NDArray<float>* input, NDArray<float>* target, NDArray<float>* result, int k);
    template int inTopKFunctor<float16>(NDArray<float16>* input, NDArray<float16>* target, NDArray<float16>* result, int k);
    template int inTopKFunctor<double>(NDArray<double>* input, NDArray<double>* target, NDArray<double>* result, int k);

}
}
}