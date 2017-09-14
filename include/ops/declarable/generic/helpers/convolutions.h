//
//
//

#ifndef LIBND4J_CONVOLUTIONS_H
#define LIBND4J_CONVOLUTIONS_H

#include <NDArray.h>

namespace nd4j {
    namespace ops {

        Nd4jIndex convsize(Nd4jIndex x, Nd4jIndex k, Nd4jIndex s, const char* vf);

        template<typename T>
        Nd4jStatus conv3D(T* output_data,
                    T alpha,
                    T* ptr_input, long nInputDepth, long nInputRows, long nInputCols,
                    T* ptr_weight, long nKernelDepth, long nKernelRows, long nKernelCols,
                    long sdepth, long srow, long scol,
                    const char *vf, const char *xc);

        template<typename T>
        Nd4jStatus conv3Dmv(NDArray<T>* r_, T beta, T alpha, NDArray<T>* t_, NDArray<T>* k_, Nd4jIndex sdepth, Nd4jIndex srow, Nd4jIndex scol, const char *vf, const char *xc);
    }
}

Nd4jIndex convsize(Nd4jIndex x, Nd4jIndex k, Nd4jIndex s, const char* vf) {
    if (*vf == 'V')
        return (x-k)/s + 1;
    else
        return (x-1)*s + k;
}

template<typename T>
Nd4jStatus nd4j::ops::conv3Dmv(NDArray<T>* r_, T beta, T alpha, NDArray<T>* t_, NDArray<T>* k_,
              Nd4jIndex sdepth, Nd4jIndex srow, Nd4jIndex scol, const char *vf, const char *xc) {

    long nInputPlane, nInputDepth, nInputRows, nInputCols;
    long nKernelDepth, nKernelRows, nKernelCols;
    long nOutputPlane, nOutputDepth, nOutputRows, nOutputCols;
    long istride0, kstride0, kstride1;
    NDArray<T> *input;
    NDArray<T> *kernel;
    T* input_data;
    T* weight_data;
    T* output_data;
    Nd4jIndex nelem;
    Nd4jIndex k, i;

    if (t_->rankOf() != 4)
        return ND4J_STATUS_BAD_DIMENSIONS;

    if (k_->rankOf() != 5)
        return ND4J_STATUS_BAD_DIMENSIONS;

    if (sdepth < 1 || srow < 1 || scol < 1)
        return ND4J_STATUS_BAD_PARAMS;

    if (!(*vf == 'V' || *vf == 'F'))
        return ND4J_STATUS_BAD_PARAMS;

    if (!(*xc == 'X' || *xc == 'C'))
        return ND4J_STATUS_BAD_PARAMS;


    if (!(k_->stridesOf()[4] == 1) || !(k_->stridesOf()[3] == k_->sizeAt(4))) {

    } else {
        kernel = k_;
    }


    nInputPlane = input->sizeAt(0);
    istride0    = input->stridesOf()[0];
    nInputDepth = input->sizeAt(1);
    nInputRows  = input->sizeAt(2);
    nInputCols  = input->sizeAt(3);

    kstride0    = kernel->stridesOf()[0];
    kstride1    = kernel->stridesOf()[1];
    nKernelDepth = kernel->sizeAt(2);
    nKernelRows = kernel->sizeAt(3);
    nKernelCols = kernel->sizeAt(4);
    nOutputPlane = kernel->sizeAt(0);

    if (kernel->sizeAt(1) != nInputPlane)
        return ND4J_STATUS_BAD_DIMENSIONS;


    if (!((nInputDepth >= nKernelDepth && nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F'))
        return ND4J_STATUS_BAD_PARAMS;

    nOutputDepth = convsize(nInputDepth, nKernelDepth, sdepth, vf);
    nOutputRows = convsize(nInputRows, nKernelRows, srow, vf);
    nOutputCols = convsize(nInputCols, nKernelCols, scol, vf);

    nelem = r_->lengthOf();

    if (nelem == 0 || beta == (T) 0.0f || nelem != r_->lengthOf()) {
        r_->assign((T) 0.0f);
    }
    else if (beta != (T) 1.0f) // stupid comparison
        r_->template applyScalar<simdOps::Multiply<T>>(beta);



    return ND4J_STATUS_OK;
}

template<typename T>
Nd4jStatus nd4j::ops::conv3D(T* output_data,
            T alpha,
            T* ptr_input, long nInputDepth, long nInputRows, long nInputCols,
            T* ptr_weight, long nKernelDepth, long nKernelRows, long nKernelCols,
            long sdepth, long srow, long scol,
            const char *vf, const char *xc) {

    if (!(*vf == 'V' || *vf == 'F'))
        return ND4J_STATUS_BAD_PARAMS;

    if (!(*xc == 'X' || *xc == 'C'))
        return ND4J_STATUS_BAD_PARAMS;


    if (*vf == 'F')
        if (*xc == 'X') {

        } else {

        }
    else
        if (*xc == 'X') {

        } else {

        }

    return ND4J_STATUS_OK;
}

#endif //LIBND4J_CONVOLUTIONS_H
