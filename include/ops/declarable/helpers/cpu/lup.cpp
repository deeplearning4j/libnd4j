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

        if (theFirst != theSecond)
        for (int i = 0; i < matrix->columns(); i++) {
            std::swap((*matrix)(theFirst, i), (*matrix)(theSecond, i));
        }
    }
    template void swapRows(NDArray<float>* matrix, int theFirst, int theSecond);
    template void swapRows(NDArray<float16>* matrix, int theFirst, int theSecond);
    template void swapRows(NDArray<double>* matrix, int theFirst, int theSecond);

    template <typename T>
    void invertLowerMatrix(NDArray<T>* inputMatrix, NDArray<T>* invertedMatrix) {
        int n = inputMatrix->rows();
        invertedMatrix->assign(T(0.0));

        for (int i = 0; i < n; i++)
            (*invertedMatrix)(i, i) = T(1.0);

        if (inputMatrix->isIdentityMatrix()) return;

        for (int i = 1; i < n; i++)
            (*invertedMatrix)(i, i - 1) = -(*inputMatrix)(i, i - 1);

        for (int i = 2; i < n; i++) {
            for (int j = i - 2; j > -1; --j) 
                for (int k = 0; k < i; k++) 
                    (*invertedMatrix)(i, j) -= (*invertedMatrix)(k, j) * (*inputMatrix)(i, k);
        }
    }

    template void invertLowerMatrix(NDArray<float>* inputMatrix, NDArray<float>* invertedMatrix);
    template void invertLowerMatrix(NDArray<float16>* inputMatrix, NDArray<float16>* invertedMatrix);
    template void invertLowerMatrix(NDArray<double>* inputMatrix, NDArray<double>* invertedMatrix);

    template <typename T>
    void invertUpperMatrix(NDArray<T>* inputMatrix, NDArray<T>* invertedMatrix) {
        int n = inputMatrix->rows();
        invertedMatrix->assign(T(0.0));

        for (int i = 0; i < n; i++)
            (*invertedMatrix)(i, i) =  T(1.0) / (*inputMatrix)(i, i);

        if (inputMatrix->isIdentityMatrix()) return;

        for (int i = 0; i < n - 1; i++)
            (*invertedMatrix)(i, i + 1) -= (*inputMatrix)(i, i + 1) / ((*inputMatrix)(i, i) * (*inputMatrix)(i + 1, i + 1));

        for (int i = n - 2; i > - 1; i--) {
            for (int j = i + 1; j < n; j++) 
                for (int k = i + 1; k < n; k++) 
                    (*invertedMatrix)(i, j) -= (*invertedMatrix)(k, j) * (*inputMatrix)(i, k) / (*inputMatrix)(i, i);
        }
    }

    template void invertUpperMatrix(NDArray<float>* inputMatrix, NDArray<float>* invertedMatrix);
    template void invertUpperMatrix(NDArray<float16>* inputMatrix, NDArray<float16>* invertedMatrix);
    template void invertUpperMatrix(NDArray<double>* inputMatrix, NDArray<double>* invertedMatrix);

    template <typename T>
    T lup(NDArray<T>* input, NDArray<T>* compound, NDArray<T>* permutation) {

        const int rowNum = input->rows();
        const int columnNum = input->columns();

        T determinant = (T)1.0;
        std::unique_ptr<NDArray<T>> compoundMatrix(new NDArray<T>(*input)); // copy
        std::unique_ptr<NDArray<T>> permutationMatrix(NDArrayFactory<T>::createUninitialized(input)); //put identity
        permutationMatrix->setIdentity();
        
        for(int i = 0; i < rowNum; i++ ) {
            T pivotValue = T(0.0);
            int pivot = -1;

            for(int rowCounter = i; rowCounter < rowNum; rowCounter++ ) {
                if(nd4j::math::nd4j_abs((*compoundMatrix)(rowCounter, i)) > pivotValue ) {
                    pivotValue = nd4j::math::nd4j_abs((*compoundMatrix)(rowCounter, i));
                    pivot = rowCounter;
                }
            }

            if( pivotValue != T(0.0) ) {
                swapRows(compoundMatrix.get(), pivot, i);
                swapRows(permutationMatrix.get(), pivot, i);
                for( int j = i + 1; j < rowNum; j++ ) {
                    (*compoundMatrix)(j, i) /= (*compoundMatrix)(i, i);
                    for( int k = i + 1; k < rowNum; k++ ) 
                        (*compoundMatrix)(j, k) -= (*compoundMatrix)(j, i) * (*compoundMatrix)(i, k);
                }

            }
        }

        for (int e = 0; e < rowNum; e++)
            determinant *= (*compoundMatrix)(e, e);

        if (compound != nullptr)
            *compound = *compoundMatrix;
        if (permutation != nullptr)
            *permutation = *permutationMatrix;

        return determinant;
    }

    template float lup(NDArray<float>* input, NDArray<float>* output, NDArray<float>* permutation);
    template float16 lup(NDArray<float16>* input, NDArray<float16>* compound, NDArray<float16>* permutation);
    template double lup(NDArray<double>* input, NDArray<double>* compound, NDArray<double>* permutation);


    template <typename T>
    int determinant(NDArray<T>* input, NDArray<T>* output) {

        int n = input->sizeAt(-1);
        int n2 = n * n;

        std::unique_ptr<NDArray<T>> matrix(new NDArray<T>({n, n})); //, block.getWorkspace());

        for (int e = 0; e < output->lengthOf(); e++) {

            for (int k = e * n2, row = 0; k < (e + 1) * n2; k++) {
                (*matrix)(row++) = (*input)(k);
            }

            (*output)(e) = lup(matrix.get(), (NDArray<T>*)nullptr, (NDArray<T>*)nullptr);
        }

        return ND4J_STATUS_OK;
    }

    template int determinant(NDArray<float>* input, NDArray<float>* output);
    template int determinant(NDArray<float16>* input, NDArray<float16>* output);
    template int determinant(NDArray<double>* input, NDArray<double>* output);

    template <typename T>
    int inverse(NDArray<T>* input, NDArray<T>* output) {

        int n = input->sizeAt(-1);
        int n2 = n * n;
        int totalCount = output->lengthOf() / n2;

        std::unique_ptr<NDArray<T>> matrix(new NDArray<T>({n, n})); //, block.getWorkspace());
        std::unique_ptr<NDArray<T>> compound(new NDArray<T>({n, n})); //, block.getWorkspace());
        std::unique_ptr<NDArray<T>> permutation(new NDArray<T>({n, n}));
        std::unique_ptr<NDArray<T>> lowerMatrix(new NDArray<T>({n, n}));
        std::unique_ptr<NDArray<T>> upperMatrix(new NDArray<T>({n, n}));

        for (int e = 0; e < totalCount; e++) {

            for (int k = e * n2, row = 0; k < (e + 1) * n2; k++) {
                (*matrix)(row++) = (*input)(k);
            }
            T det = lup(matrix.get(), compound.get(), permutation.get());

            if (nd4j::math::nd4j_abs(det) < T(0.0000001)) {
                nd4j_printf("matrix_inverse: The matrix %i has no inverse due determinant is %lf. Quiting...\n", e, det);
                matrix->printIndexedBuffer("Wrong matrix");
                ND4J_STATUS_VALIDATION;
            }
            lowerMatrix->setIdentity(); // set up U to identity matrix
            for (int k = 1; k < n; k++) {  // and then put all values under main diagonal on to it
                for (int j = 0; j < k; j++)
                    (*lowerMatrix)(k, j) = (*compound)(k, j);
            }
            upperMatrix->setIdentity(); // set up U to identity matrix
            for (int k = 0; k < n; k++) {  // and then put all values under main diagonal on to it
                for (int j = k; j < n; j++)
                    (*upperMatrix)(k, j) = (*compound)(k, j);
            }

            invertUpperMatrix(upperMatrix.get(), matrix.get());
            invertLowerMatrix(lowerMatrix.get(), upperMatrix.get());

            nd4j::NDArrayFactory<T>::mmulHelper(matrix.get(), upperMatrix.get(), compound.get(), T(1.0f), T(0.0f));
            nd4j::NDArrayFactory<T>::mmulHelper(compound.get(), permutation.get(), matrix.get(), T(1.0f), T(0.0f));

            for (int k = e * n2, row = 0; k < (e + 1) * n2; k++) {
                (*output)(k) = (*matrix)(row++);
            }
            matrix->assign((T)0.0);
        }

        return ND4J_STATUS_OK;
    }

    template int inverse(NDArray<float>* input, NDArray<float>* output);
    template int inverse(NDArray<float16>* input, NDArray<float16>* output);
    template int inverse(NDArray<double>* input, NDArray<double>* output);
}
}
}