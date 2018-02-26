//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/top_k.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T> 
    void swapRows(NDArray<T>* matrix, int theFirst, int theSecond) {
        for (int i = 0; i = matrix->columns(); i++) {
            std::swap((*matrix)(theFirst, i), (*matrix)(theSecond, i));
        }
    }
    template void swapRows(NDArray<float>* matrix, int theFirst, int theSecond);
    template void swapRows(NDArray<float16>* matrix, int theFirst, int theSecond);
    template void swapRows(NDArray<double>* matrix, int theFirst, int theSecond);

    template <typename T>
    T lup(NDArray<T>* input, NDArray<T>* compound, NDArray<T>* permutation) {

        const int n = input->rows();
        T determinant = (T)1.0;
        std::unique_ptr<NDArray<T>> compoundMatrix(new NDArray<T>(*input)); // copy
        std::unique_ptr<NDArray<T>> permutationMatrix(NDArrayFactory<T>::createUninitialized(input)); //put identity
        permutationMatrix->setIdentity();
    
        for(int i = 0; i < n; i++ ) {
            //поиск опорного элемента
            double pivotValue = 0;
            int pivot = -1;
            for( int row = i; row < n; row++ ) {
                if( fabs((*compoundMatrix)(row, i)) > pivotValue ) {
                    pivotValue = fabs((*compoundMatrix)(row, i));
                    pivot = row;
                }
            }
    
            if( pivotValue != 0 ) {
               //меняем местами i-ю строку и строку с опорным элементом
                swapRows(compoundMatrix.get(), pivot, i);
                swapRows(permutationMatrix.get(), pivot, i);
                for( int j = i+1; j < n; j++ ) {
                    (*compoundMatrix)(j, i) /= (*compoundMatrix)(i, i);
                    for( int k = i + 1; k < n; k++ ) 
                        (*compoundMatrix)(j , k) -= (*compoundMatrix)(j, i) * (*compoundMatrix)(i, k);
                }
            }
        }
    
        for (int i = 0; i < n; i++)
            determinant *= (*compound)(i, i);

        if (compound != nullptr)
            *compound = *compoundMatrix;
        if (permutation != nullptr)
            *permutation = *permutationMatrix;

        return determinant;
    }

    template float lup(NDArray<float>* input, NDArray<float>* output, NDArray<float>* permutation);
    template float16 lup(NDArray<float16>* input, NDArray<float16>* compound, NDArray<float16>* permutation);
    template double lup(NDArray<double>* input, NDArray<double>* compound, NDArray<double>* permutation);
}
}
}