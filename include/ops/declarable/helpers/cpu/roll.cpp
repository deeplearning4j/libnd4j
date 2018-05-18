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
        if (!inplace)
            output->assign(input);

        int fullLen = source->lengthOf();
        int actualShift = shift; // % fullLen; // shift already non-negative then
        if (actualShift < 0) {
            actualShift -= fullLen * (actualShift / fullLen - 1);
        }
        else
            actualShift %= fullLen;

        if (actualShift) {

#pragma omp parallel for if (actualShift > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 0; e < actualShift; ++e) {
                int sourceIndex = fullLen - actualShift + e;
                nd4j::math::nd4j_swap((*output)(e), (*output)(sourceIndex));
            }

            bool again = actualShift < fullLen - actualShift;
            int k = 1;
            while(again) {
                for (int e = 0; e < actualShift; ++e) {
                    nd4j::math::nd4j_swap((*output)(fullLen - k * actualShift + e), (*output)(fullLen - (k + 1) * actualShift + e));
                }
                k++;
                again = actualShift < fullLen - (k + 1) * actualShift;
            }
            int rest = fullLen - k * actualShift - actualShift;
            again = rest > 0;
            k = 0;
            while(again){
                for (int e = 0; e < rest; ++e) {
                    nd4j::math::nd4j_swap((*output)(actualShift + (k + 1)* rest + e), (*output)(actualShift + k * rest + e));
                }
                k++;
                again = (2 * actualShift > (actualShift + k * rest)) && (actualShift + k * rest < fullLen);
            }
        }
    }

    template <typename T>
    void rollFunctorFull(NDArray<T>* input, NDArray<T>* output, int shift, std::vector<int> const& axes, bool inplace){

        if (!inplace)
            output->assign(input);

        NDArray<T>* source = input;
        for (int axe: axes) {
            if (axe == source->rankOf() - 1) {// last dimension
                std::unique_ptr<ResultSet<T>> listOfTensors(NDArrayFactory<T>::allTensorsAlongDimension(source, {axe}));
                std::unique_ptr<ResultSet<T>> listOfOutTensors(NDArrayFactory<T>::allTensorsAlongDimension(output, {axe}));
                int fullLen = listOfTensors->size();
                int theShift = shift;
                if (theShift > 0) {
                    theShift %= fullLen;
                }
                else {
                        theShift -= fullLen * (theShift / fullLen - 1);
                }
                for (int k = 0; k < fullLen; k++) {
                    rollFunctorLinear(listOfTensors->at(k), listOfOutTensors->at(k), theShift, true);
                }
            }
            else {
                std::vector<int> dims(source->rankOf() - axe - 1);
                for (int i = 0; i < dims.size(); ++i)
                    dims[i] = axe + 1 + i;

                std::unique_ptr<ResultSet<T>> listOfTensors(NDArrayFactory<T>::allTensorsAlongDimension(source, {dims}));
                std::unique_ptr<ResultSet<T>> listOfOutTensors(NDArrayFactory<T>::allTensorsAlongDimension(output, {dims}));
            
                int fullLen = listOfTensors->size();
                int sizeAt = input->sizeAt(axe);

                int theShift = shift;

                if (theShift > 0) {
                    theShift %= sizeAt;
                }
                else {
                    theShift -= sizeAt * (theShift / sizeAt - 1);
                }

                if (theShift) {
                    for (int dim = 0; dim < fullLen / sizeAt; ++dim) {
                        for (int e = theShift; e < sizeAt - theShift; ++e) {
                            NDArray<T>* sourceM = listOfTensors->at(dim * sizeAt + e - theShift);
                            NDArray<T>* targetM = listOfOutTensors->at(dim * sizeAt + e);
                            sourceM->swapUnsafe(*targetM);
                        }
    
                        for (int e = 0; e < theShift; ++e) {
                            int sourceIndex = dim * sizeAt + sizeAt - theShift + e;
                            NDArray<T>* sourceM = listOfTensors->at(sourceIndex);
                            NDArray<T>* targetM = listOfOutTensors->at(dim * sizeAt + e);
    
                            sourceM->swapUnsafe(*targetM);
                        }
                    }
                }
            }
            if (!inplace)
                source = output;
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