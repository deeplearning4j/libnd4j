//
// 3D convolutions are based on pytorch - https://github.com/pytorch/pytorch
//

#ifndef LIBND4J_CONVO_OPS_H
#define LIBND4J_CONVO_OPS_H

#include <op_boilerplate.h>
#include <memory>
#include <iomanip>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/OpRegistrator.h>
#include <declarable/generic/helpers/convolutions.h>



namespace nd4j {
    namespace ops {

template <typename T>
void simpleMmul(const NDArray<T>& arr1, const NDArray<T>& arr2, NDArray<T>& result) {

    if(arr1.rankOf() !=2 || arr2.rankOf() !=2 || result.rankOf() !=2 )
        throw "simpleMmul: input array must have rank = 2 !";

    if(arr1.sizeAt(1) != arr2.sizeAt(0))
        throw "simpleMmul: number of columns of first array must be equal to number of rows of second array !";

     // multiplication
#pragma omp parallel for collapse(2) schedule(guided)        
    for(int i = 0; i < arr1.sizeAt(0); ++i)
        for(int j = 0; j < arr2.sizeAt(1); ++j) {
            result(i,j) = 0.;                                   // initializing elements of result to 0
            for(int k = 0; k < arr1.sizeAt(1); ++k)
                result(i,j) += arr1(i,k) * arr2(k,j);
        }
}





        CUSTOM_OP_IMPL(conv2d, 2, 1, false, 0, 10) {
            // basically im2col + gemm
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* weights = INPUT_VARIABLE(1);
            NDArray<T>* bias = nullptr;


            REQUIRE_TRUE(input->rankOf() == 4, 0, "Conv2D: input should be 4D NDArray, but got %i instead", input->rankOf());
            REQUIRE_TRUE(weights->rankOf() == 4, 0, "Conv2D: weights should be 4D NDArray, but got %i instead", weights->rankOf());

            if (block.width() == 3)
                bias = INPUT_VARIABLE(2);

            const int kH = INT_ARG(0);
            const int kW = INT_ARG(1);
            const int sH = INT_ARG(2);
            const int sW = INT_ARG(3);
            int pH = INT_ARG(4);
            int pW = INT_ARG(5);
            int dH = INT_ARG(6);
            int dW = INT_ARG(7);
            const bool isSameMode = INT_ARG(8) != 0;
            bool isNCHW = true;
            if (block.getIArguments()->size() > 9)
                isNCHW = INT_ARG(9) == 0;
            const double zeroPadVal = 0.0;

            if (!isNCHW) {
                input = input->permute({0, 3, 1, 2});
                //input = input->dup('c');
                auto weightsT = weights->permute({3, 2, 0, 1});
                weights = weightsT->dup('c');

                delete weightsT;

                //input->printShapeInfo("new input");
                //weights->printShapeInfo("new shape");
            }

            int bS = input->sizeAt(0);

            const int oC = weights->sizeAt(0);
            const int iC = weights->sizeAt(1);
            const int iH = input->shapeOf()[2];
            const int iW = input->shapeOf()[3];

            REQUIRE_TRUE(weights->sizeAt(2) == kH, 0, "Conv2D: weights dim 2 should be equal to %i, but got %i instead. Not a NCHW?", kH, weights->sizeAt(2));
            REQUIRE_TRUE(weights->sizeAt(3) == kW, 0, "Conv2D: weights dim 3 should be equal to %i, but got %i instead. Not a NCHW?", kW, weights->sizeAt(3));
            REQUIRE_TRUE(iC == input->sizeAt(1), 0, "Conv2D: weights dim 1 should be equal to number of input channels. But got %i vs %i. Not a NCHW?", weights->sizeAt(1), input->sizeAt(1))

            if (bias != nullptr) {
                REQUIRE_TRUE(weights->sizeAt(0) == bias->lengthOf(), 0, "Conv2D: bias length should be equal to outChannels, but got %i instead", bias->lengthOf());
            }

            int oH = 0;
            int oW = 0;


            REQUIRE_TRUE(weights->shapeOf()[2] == kH && weights->shapeOf()[3] == kW, 0, "Kernels should have dimensions of [%i, %i], but got [%i, %i] instead", kH, kW, weights->sizeAt(2), weights->sizeAt(3));

            ConvolutionUtils<T>::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

            if (isSameMode)
                ConvolutionUtils<T>::_calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

            NDArray<T>* output = OUTPUT_VARIABLE(0);

            Nd4jIndex prod = bS * oC * oH * oW;

            if (isNCHW) {
                REQUIRE_TRUE(
                        output->sizeAt(0) == bS && output->sizeAt(1) == oC && output->sizeAt(2) == oH &&
                        output->sizeAt(3) == oW, 0,
                        "Expected output shape is [%i, %i, %i, %i] but got [%i, %i, %i, %i] instead", bS,
                        oC, oH, oW, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2), output->sizeAt(3));
            } else {
                REQUIRE_TRUE(
                        output->sizeAt(0) == bS && output->sizeAt(1) == oH &&
                        output->sizeAt(2) == oW && output->sizeAt(3) == oC , 0,
                        "Expected output shape is [%i, %i, %i, %i] but got [%i, %i, %i, %i] instead", bS,
                        oC, oH, oW, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2), output->sizeAt(3));
            }
            REQUIRE_TRUE(output->lengthOf() == prod, 0, "Z should have total length of %i, but got %i instead", prod, output->lengthOf());

            std::unique_ptr<NDArray<T>> col(new NDArray<T>('c', {bS, oH, oW, iC, kH, kW}));
            std::unique_ptr<NDArray<T>> col2(col.get()->permute({0, 3, 4, 5, 1, 2}));

            std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW, isSameMode ? (T) 1.0f : (T) 0.0f, (T)zeroPadVal});

            input->template applyTransform<simdOps::Im2col<T>>(col2.get(), extrasIm2Col.data());

            std::unique_ptr<NDArray<T>> im2col2d(col->reshape('c', {bS * oH * oW, iC * kH * kW}));
            std::unique_ptr<NDArray<T>> permutedW(weights->permute({3, 2, 1, 0}));
            std::unique_ptr<NDArray<T>> reshapedW(permutedW.get()->reshape('f', {kW * kH * iC, oC}));

            //output->reshapei('f', {im2col2d.get()->rows(), reshapedW.get()->columns()});
            output->enforce({im2col2d.get()->rows(), reshapedW.get()->columns()}, 'f');

            NDArrayFactory<T>::mmulHelper(im2col2d.get(), reshapedW.get(), output, 1.0, 0.0);

            // bias addition is optional
            if (bias != nullptr) {
                if (!bias->isRowVector())
                    bias->transposei();

                // FIXME: do we really want transposei() above?
                output->addiRowVector(bias);
            }

            output->reshapei('f', {oW, oH, bS, oC});

            output->permutei({2, 3, 1, 0});

            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                output->printShapeInfo("Conv2D result shape");

            STORE_RESULT(*output);

            if (!isNCHW) {
                delete input;
                delete weights;

                output->permutei({0, 2, 3, 1});
//                auto f = output->dup('c');
                //f->printShapeInfo("final shape");
                //OVERWRITE_RESULT(f);

  //              f->printIndexedBuffer("conv2d output");
            }

            return ND4J_STATUS_OK;
        }
        
        DECLARE_SHAPE_FN(conv2d) {
            auto inShape = inputShape->at(0);
            auto wShape = inputShape->at(1);

            const int kH = INT_ARG(0);
            const int kW = INT_ARG(1);
            const int sH = INT_ARG(2);
            const int sW = INT_ARG(3);
            int pH = INT_ARG(4);
            int pW = INT_ARG(5);
            const int dH = INT_ARG(6);
            const int dW = INT_ARG(7);
            const bool isSameMode = INT_ARG(8) != 0;

            bool isNCHW = true;
            if (block.getIArguments()->size() > 9)
                isNCHW = INT_ARG(9) == 0;

            int oH = 0;
            int oW = 0;

            const int bS = shape::sizeAt(inShape, 0);
            const int oC = isNCHW ? shape::sizeAt(wShape, 0) : shape::sizeAt(wShape, 3);
            const int iH = isNCHW ? shape::sizeAt(inShape, 2) : shape::sizeAt(inShape, 1);
            const int iW = isNCHW ? shape::sizeAt(inShape, 3) : shape::sizeAt(inShape, 2);

            ConvolutionUtils<T>::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

            if (isSameMode) {
                ConvolutionUtils<T>::_calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);
            }

            //z = Shape.newShapeNoCopy(z, new int[] {outW, outH, miniBatch, oC}, true);
            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(4), int);
            if (isNCHW) {
                std::vector<int> shape({bS, oC, oH, oW});
                shape::shapeBuffer(4, shape.data(), newShape);
            } else {
                std::vector<int> shape({bS, oH, oW, oC});
                shape::shapeBuffer(4, shape.data(), newShape);
            }

            return SHAPELIST(newShape);
        }

////////////////////////////////////////////////////////////////////////// 
CUSTOM_OP_IMPL(conv2d_bp, 3, 2, false, 0, 9) {
    
    NDArray<T> *input   = INPUT_VARIABLE(0);                                                // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
    NDArray<T> *weights = INPUT_VARIABLE(1);                                                // [kH, kW, iC, oC] (NDHWC) or [oC, iC, kH, kW] (NCDHW)
    NDArray<T> *bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
    NDArray<T> *gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next
    
    NDArray<T> *gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW), epsilon
    NDArray<T> *gradW = OUTPUT_VARIABLE(1);                                                 // [kH, kW, iC, oC] (NDHWC) or [oC, iC, kH, kW] (NCDHW)
    NDArray<T> *gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]
    
    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM CONV2D_BP OP: rank of input array must be equal to 4 !");
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM CONV2D_BP OP: rank of weights array must be equal to 4 !");
    REQUIRE_TRUE(gradO->rankOf() == 4, 0, "CUSTOM CONV2D_BP OP: rank of gradO array must be equal to 4 !");
                                     
    int kH = INT_ARG(0);                                                        // filter(kernel) height
    int kW = INT_ARG(1);                                                        // filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int isValidMode = INT_ARG(8);                                               // 0-SAME, 1-VALID
    int isNCHW  = block.getIArguments()->size() > 9 ? INT_ARG(9) : 0;           // 0-NHWC, 1-NCHW    

    if(!isNCHW) {
        input   = input->permute({0, 3, 1, 2});                                 // [bS, iH, iW, iC] -> [bS, iC, iH, iW]                        
        gradI   = gradI->permute({0, 3, 1, 2});                                 // [bS, iH, iW, iC] -> [bS, iC, iH, iW]        
        weights = weights->permute({2, 0, 1, 3});                               // [kH, kW, iC, oC] -> [iC, kH, kW, oC]         
        gradW   = gradW->permute({2, 0, 1, 3});                                 // [kH, kW, iC, oC] -> [iC, kH, kW, oC]                 
    }
    else {
        gradO   = gradO->permute({0, 2, 3, 1});                                 // [bS, oC, oH, oW] -> [bS, oH, oW, oC]
        weights = weights->permute({1, 2, 3, 0});                               // [oC, iC, kH, kW] -> [iC, kH, kW, oC]
        gradW   = gradW->permute({1, 2, 3, 0});                                 // [oC, iC, kH, kW] -> [iC, kH, kW, oC]

        // gradO->streamline('c');
    }

    int bS = input->sizeAt(0);           // batch size
    int iC = input->sizeAt(1);           // input channels        
    int iH = input->sizeAt(2);           // input height
    int iW = input->sizeAt(3);           // input width
    int oC = weights->sizeAt(3);         // output channels    
    int oH = gradO->sizeAt(1);           // output height
    int oW = gradO->sizeAt(2);           // output width    

    int trueoH, trueoW;          // correct output height, width
    ConvolutionUtils<T>::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, !isValidMode);

    REQUIRE_TRUE(gradO->sizeAt(0)==bS   && gradO->sizeAt(1)==trueoH && gradO->sizeAt(2)==trueoW && gradO->sizeAt(3)==oC, 0, "CUSTOM CONV2D_BP OP: wrong shape of gradient_output (next epsilon) array !");    
    REQUIRE_TRUE(weights->sizeAt(0)==iC && weights->sizeAt(1)==kH   && weights->sizeAt(2)==kW, 0, "CUSTOM CONV2D_BP OP: wrong shape of weights array !");
    if(bias)
        REQUIRE_TRUE(bias->rankOf()==1 && bias->lengthOf()==oC, 0, "CUSTOM CONV2D_BP OP: wrong shape of biases array !");

    if(!isValidMode)                       // SAME        
        ConvolutionUtils<T>::_calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray<T>  columns(input->ordering(), {iC, kH, kW, bS, oH, oW});        
    NDArray<T>* columnsPermuted = columns.permute({3, 0, 1, 2, 4, 5});                                 // [iC, kH, kW, bS, oH, oW] -> [bS, iC, kH, kW, oH, oW]
    NDArray<T>* columnsReshaped = columns.reshape(columns.ordering(), {iC*kH*kW, bS*oH*oW});
    NDArray<T>* gradWreshaped   = gradW->reshape(gradW->ordering(),{iC*kH*kW, oC});    
    NDArray<T>* weightsReshaped = weights->reshape(weights->ordering(), {iC*kH*kW, oC});    
    NDArray<T>* gradOreshaped   = gradO->reshape(gradO->ordering(),{bS*oH*oW, oC});    
    NDArray<T>* gradOreshapedT  = gradOreshaped->transpose();                                           // [bS*oH*oW, oC] -> [oC, bS*oH*oW]

    // ----- calculation of gradW and gradB ----- //            
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});
    input->template applyTransform<simdOps::Im2col<T>>(columnsPermuted, extrasIm2Col.data());          // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]    
    NDArrayFactory<T>::mmulHelper(columnsReshaped, gradOreshaped, gradWreshaped, 1.0, 0.0);            // [iC*kW*kH, bS*oH*oW] x [bS*oH*oW, oC] = [iC*kH*kW, oC]

    if(gradB) {
        NDArray<T>* sum = gradOreshaped->sum({0});                  // sum over bS*oH*oW
        gradB->assign(sum);
        delete sum;
    }

    //----- calculation of gradI -----//            
    NDArrayFactory<T>::mmulHelper(weightsReshaped, gradOreshapedT, columnsReshaped, 1.0, 0.0);             // [iC*kH*kW, oC] x [oC, bS*oH*oW] = [iC*kW*kH, bS*oH*oW]
    std::vector<T> extrasCol2Im({(T) sH, (T) sW, (T) pH, (T) pW, (T) iH, (T) iW, (T) dH, (T) dW});
    columnsPermuted->template applyTransform<simdOps::Col2Im<T>>(gradI, extrasCol2Im.data());               // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]

    //----- assign array having separate new shape (caused by permute+reshape ops) to output gradW -----///
    gradW->assign(gradWreshaped);

    //----- clean dynamically allocated memory -----//
    delete gradOreshapedT;
    delete columnsPermuted;
    delete columnsReshaped;
    delete gradWreshaped;
    delete weightsReshaped;
    delete gradOreshaped;    
    delete weights;
    delete gradW;

   
    if(!isNCHW) {        
        delete input;        
        delete gradI;
    }
    else {
        delete gradO;              
            
    }
    
    return Status::OK();
}



DECLARE_SHAPE_FN(conv2d_bp) {

    int* inputShapeInfo   = inputShape->at(0);
    int* weightsShapeInfo = inputShape->at(1);
    int* biasShapeInfo    = block.width() > 3 ? inputShape->at(2) : nullptr;  

    int* gradIshapeInfo(nullptr), *gradWshapeInfo(nullptr);
    COPY_SHAPE(inputShapeInfo, gradIshapeInfo);
    COPY_SHAPE(weightsShapeInfo, gradWshapeInfo);

    if(biasShapeInfo) {
        int* gradBshapeInfo(nullptr);
        COPY_SHAPE(biasShapeInfo, gradBshapeInfo);
        return SHAPELIST(gradIshapeInfo, gradWshapeInfo, gradBshapeInfo);
    }     

    return SHAPELIST(gradIshapeInfo, gradWshapeInfo);        
}


}
}

#endif //LIBND4J_CONVO_OPS_H
