//
//  @author raver119@gmail.com
//

#include <helpers/BlasHelper.h>
namespace nd4j {
    BlasHelper* BlasHelper::getInstance() {
        if (_instance == 0)
            _instance = new BlasHelper();
        return _instance;
    }


    void BlasHelper::initializeFunctions(Nd4jPointer *functions) {
        this->cblasSgemv = (CblasSgemv)functions[0];
        this->cblasDgemv = (CblasDgemv)functions[1];
        this->cblasSgemm = (CblasSgemm)functions[2];
        this->cblasDgemm = (CblasDgemm)functions[3];
        this->cblasSgemmBatch = (CblasSgemmBatch)functions[4];
        this->cblasDgemmBatch = (CblasDgemmBatch)functions[5];

        this->cublasSgemv = (CublasSgemv)functions[6];
        this->cublasDgemv = (CublasDgemv)functions[7];
        this->cublasHgemm = (CublasHgemm)functions[8];
        this->cublasSgemm = (CublasSgemm)functions[9];
        this->cublasDgemm = (CublasDgemm)functions[10];
        this->cublasSgemmEx = (CublasSgemmEx)functions[11];
        this->cublasHgemmBatched = (CublasHgemmBatched)functions[12];
        this->cublasSgemmBatched = (CublasSgemmBatched)functions[13];
        this->cublasDgemmBatched = (CublasDgemmBatched)functions[14];
    }


    template <>
    bool BlasHelper::hasGEMV<float>() {
        return this->cblasSgemv != nullptr;
    }

    template <>
    bool BlasHelper::hasGEMV<double>() {
        return this->cblasDgemv != nullptr;
    }

    template <>
    bool BlasHelper::hasGEMV<float16>() {
        return false;
    }

    template <>
    bool BlasHelper::hasGEMM<float>() {
        return this->cblasSgemm != nullptr;
    }

    template <>
    bool BlasHelper::hasGEMM<double>() {
        return this->cblasDgemm != nullptr;
    }

    template <>
    bool BlasHelper::hasGEMM<float16>() {
        return false;
    }

    template <>
    bool BlasHelper::hasBatchedGEMM<float>() {
        return this->cblasSgemmBatch != nullptr;
    }

    template <>
    bool BlasHelper::hasBatchedGEMM<double>() {
        return this->cblasDgemmBatch != nullptr;
    }

    template <>
    bool BlasHelper::hasBatchedGEMM<float16>() {
        return false;
    }


    CblasSgemv BlasHelper::sgemv() {
        return this->cblasSgemv;
    }
    CblasDgemv BlasHelper::dgemv() {
        return this->cblasDgemv;
    }

    CblasSgemm BlasHelper::sgemm() {
        return this->cblasSgemm;
    }

    CblasDgemm BlasHelper::dgemm() {
        return this->cblasDgemm;
    }

    BlasHelper* BlasHelper::_instance = 0;
}