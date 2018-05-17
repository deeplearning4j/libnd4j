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

//        if (!inplace)
//            output->assign(input);

        NDArray<T>* source = input; //output;
        for (int axe: axes) {
            nd4j_printf("Axe %i:\n", axe);
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
                nd4j_printf("Axe %i; Total rows: %i; Shift: %i\n", axe, fullLen, theShift);

                for (int k = 0; k < fullLen; k++) {
                    listOfTensors->at(k)->printIndexedBuffer("Tensor at last");
                    rollFunctorLinear(listOfTensors->at(k), listOfOutTensors->at(k), theShift);
                }
                nd4j_printf("Axe %i: All done\n", axe);
                output->printIndexedBuffer("The result at ");
            }
            else {
                std::vector<int> dims(source->rankOf() - axe - 1);
                for (int i = 0; i < dims.size(); ++i)
                    dims[i] = axe + 1 + i;
                std::unique_ptr<ResultSet<T>> listOfTensors(NDArrayFactory<T>::allTensorsAlongDimension(source, {dims}));
                std::unique_ptr<ResultSet<T>> listOfOutTensors(NDArrayFactory<T>::allTensorsAlongDimension(output, {dims}));
            
                int fullLen = listOfTensors->size();
                int sizeAt = input->sizeAt(axe);
                nd4j_printf("Total: %i; Dim %i size is %i and should be split on %i(%i).\n", fullLen, axe, sizeAt, fullLen / sizeAt, fullLen % sizeAt);
                for (int k = 0; k < fullLen; k++) {
                    listOfTensors->at(k)->printIndexedBuffer("Tensor at ");
                }
                //if (axe == 0) continue;
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
                        nd4j_printf("(first) Exchange %i <--> %i.\n", dim * sizeAt + e - shift, dim * sizeAt + e);
                        NDArray<T>* sourceM = listOfTensors->at(dim * sizeAt + e - theShift);
                        NDArray<T>* targetM = listOfOutTensors->at(dim * sizeAt + e);
                        sourceM->swapUnsafe(*targetM);
                        //listOfOutTensors->at(e)->assign(listOfTensors->at(e - theShift));
                    }

                    nd4j_printf("Than\n", "");

                    for (int e = 0; e < theShift; ++e) {
                        int sourceIndex = dim * sizeAt + sizeAt - theShift + e;
                        nd4j_printf("(last) Exchange %i <--> %i.\n", dim * sizeAt + e, sourceIndex);
                        NDArray<T>* sourceM = listOfTensors->at(sourceIndex);
                        NDArray<T>* targetM = listOfOutTensors->at(dim * sizeAt + e);
                        //listOfOutTensors->at(e)->assign(listOfTensors->at(sourceIndex));
                        sourceM->swapUnsafe(*targetM);
                    }

                    for (int k = 0; k < fullLen; k++) {
                        listOfTensors->at(k)->printIndexedBuffer("Tensor after at ");
                    }

                    }
                }
                    //rollFunctorLinear(listOfTensors->at(k), listOfOutTensors->at(k), theShift);
            }
                // dims
            //delete source;
            source = output;
        }
        //delete source;
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