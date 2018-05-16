//
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/roll.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void rollFunctorLinear(NDArray<T>* input, NDArray<T>* output, int shift, bool inplace){
        NDArray<T>* source = input;
        int fullLen = output->lengthOf();
        int actualShift = shift; // % fullLen; // shift already non-negative then
        if (actualShift < 0) {
//            if (actualShift + fullLen > 0) actualShift += fullLen;
//            else {
                actualShift -= fullLen * (actualShift / fullLen - 1);
//            }
        }
        else
            actualShift %= fullLen;

        if (actualShift) {
            if (inplace)
                source = input->dup();

#pragma omp parallel for 
            for (int e = 0; e < actualShift; ++e) {
                int sourceIndex = fullLen - actualShift + e;
                (*output)(e) = (*source)(sourceIndex);
            }
    
#pragma omp parallel for 
            for (int e = actualShift; e < fullLen; ++e) {
                (*output)(e) = (*source)(e - actualShift);
            }
            if (inplace)
                delete source;
        }
        else {// no any changes
            if (!inplace) output->assign(input);
        }
    }

    template <typename T>
    void rollFunctorFull(NDArray<T>* input, NDArray<T>* output, int shift, std::vector<int> const& axes, bool inplace){
        if (inplace) {
        }
        else {
        NDArray<T>* source = input;
        for (int axe: axes) {
            std::unique_ptr<ResultSet<T>> listOfTensors(NDArrayFactory<T>::allTensorsAlongDimension(source, {axe}));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(NDArrayFactory<T>::allTensorsAlongDimension(output, {axe}));

            int fullLen = listOfTensors->size();
            for (int k = 0; k < fullLen; k++) {
                listOfTensors->at(k)->printIndexedBuffer("Tensor at 0");
            }
            if (shift > 0) {
                shift %= fullLen;
            }
            else {
                shift -= fullLen * (shift / fullLen - 1);
            }
            if (shift) {
                for (int e = 0; e < shift; ++e) {
                    int sourceIndex = fullLen - shift + e;
                    listOfOutTensors->at(e)->assign(listOfTensors->at(sourceIndex));
                }

                for (int e = shift; e < fullLen; ++e) {
                    listOfOutTensors->at(e)->assign(listOfTensors->at(e - shift));
                }
            }
/*            else if (shift < 0) {
                shift %= fullLen;
                for (int e = 0; e < fullLen + shift; ++e) {
                    listOfOutTensors->at(e)->assign(listOfTensors->at(e - shift));
                }
                for (int e = fullLen + shift; e < fullLen; ++e) {
                    (*output)(e) = (*input)(e - fullLen - shift);
                    listOfOutTensors->at(e)->assign(listOfTensors->at(e - fullLen - shift));
                }
            }*/
            else
                output->assign(input);
                source = output;
        }
        }
    }

    template void rollFunctorLinear(NDArray<float>*   input, NDArray<float>*   output, int shift, bool inplace);
    template void rollFunctorLinear(NDArray<float16>* input, NDArray<float16>* output, int shift, bool inplace);
    template void rollFunctorLinear(NDArray<double>*  input, NDArray<double>*  output, int shift, bool inplace);
    template void rollFunctorFull(NDArray<float>*   input, NDArray<float>* axisVector, int shift, std::vector<int> const& axes, bool inplace);
    template void rollFunctorFull(NDArray<float16>* input, NDArray<float16>* axisVector, int shift, std::vector<int> const& axes, bool inplace);
    template void rollFunctorFull(NDArray<double>*  input, NDArray<double>* axisVector, int shift, std::vector<int> const& axes, bool inplace);
}
}
}