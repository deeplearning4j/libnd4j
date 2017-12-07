// //
// // Created by Yurii Shyrma on 06.12.2017.
// //

// #include <ops/declarable/CustomOperations.h>


// namespace nd4j {
//     namespace ops {


// //////////////////////////////////////////////////////////////////////////
// // Construct an identity matrix, or a batch of matrices
// // for detailed explanations please take a look on web page: https://www.tensorflow.org/api_docs/python/tf/eye
// CUSTOM_OP_IMPL(eye, 0, 1, false, 0, 0) {

// 	NDArray<T>* input   = INPUT_VARIABLE(0);
// 	NDArray<T>* indices = INPUT_VARIABLE(1);
// 	NDArray<T>* output  = OUTPUT_VARIABLE(0);

// 	int axis = block.getIArguments()->at(0);
    
    
//     // first case: indices consist of only one scalar
//    	if(indices->isScalar()) {
//    		std::vector<int> dimensions = ShapeUtils<T>::evalDimsToExclude(input->rankOf(), {axis});
//    		shape::TAD tad(input->getShapeInfo(), dimensions.data(), dimensions.size());
//    		tad.createTadOnlyShapeInfo();
//     	tad.createOffsets();
//     	NDArray<T> tadArr(input->getBuffer() + tad.tadOffsets[(int)indices->getScalar(0)], tad.tadOnlyShapeInfo);
//     	output->assign(&tadArr);
//    	}
//    	// second case: indices is vector
//    	else if(indices->isVector()) {   	
//    		ResultSet<T>* listOut = NDArrayFactory<T>::allTensorsAlongDimension(output, ShapeUtils<T>::evalDimsToExclude(output->rankOf(), {axis}));
//    		ResultSet<T>* listIn  = NDArrayFactory<T>::allTensorsAlongDimension(input,  ShapeUtils<T>::evalDimsToExclude(input->rankOf(),  {axis}));
//    		for(int i = 0; i < listOut->size(); ++i)
//    			listOut->at(i)->assign(listIn->at((int)indices->getIndexedScalar(i)));
//    		delete listOut;
//    		delete listIn;
//    	}
//    	// third case: indices is usual n-dim array
//    	else {
//    		std::vector<int> dimsOut(indices->rankOf());
//    		std::iota(dimsOut.begin(), dimsOut.end(), axis);   // fill with axis, axis+1, ... indices->rankOf()-1
//    		std::vector<int> temp1 = ShapeUtils<T>::evalDimsToExclude(output->rankOf(), dimsOut);
//    		std::vector<int> temp2 = ShapeUtils<T>::evalDimsToExclude(input->rankOf(),  {axis});
//    		ResultSet<T>* listOut = NDArrayFactory<T>::allTensorsAlongDimension(output, temp1);
//    		ResultSet<T>* listIn = NDArrayFactory<T>::allTensorsAlongDimension(input,  temp2 );
//    		for(int i = 0; i < listOut->size(); ++i)
//    			listOut->at(i)->assign(listIn->at((int)indices->getIndexedScalar(i)));
//    		delete listOut;
//    		delete listIn;
//    	}

//     STORE_RESULT(*output);	

//     return ND4J_STATUS_OK;
// }


// DECLARE_SHAPE_FN(eye) {

//     std::vector<int>* iArgs = block.getIArguments();
//     int* outputShapeInfo = nullptr;

//     switch (block.width()) {
//         case 0: 
//             if(iArgs->empty())
//                 throw "CUSTOM_OP eye: neither input array nor vector with integer arguments is present !";
//             for(const auto& arg : *iArgs)
//                 if(arg <= 0)
//                     throw "CUSTOM_OP eye: some of integer arguments is <= 0 !";
//                 int sizeOfIargs = iArgs->size();                
//                 if(sizeOfIargs == 1) {                    
//                     ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(2), int);
//                     outputShapeInfo[0] = 2;
//                     outputShapeInfo[1] = outputShapeInfo[2] = *iArgs[0];
//                 }
//                 else {
//                     ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(sizeOfIargs), int);
//                     outputShapeInfo[0] = sizeOfIargs;
//                     for(int i = 0; i < sizeOfIargs; ++i)
//                         outputShapeInfo[i+1] = *iArgs[i];
//                 }
//             break;
//         case 1:             // REFLECT mode                 
//             for(int j = 1;  j <= leftOffset; ++j)                                               // fill firstly left side 
//                 subArrOut.putIndexedScalar(leftOffset - j, subArrIn.getIndexedScalar(j));                       
//             for(int j = 0; j < subArrIn.lengthOf(); ++j)                                        // fill middle
//                 subArrOut.putIndexedScalar(leftOffset + j, subArrIn.getIndexedScalar(j));                   
//             for(int j = (subArrOut.lengthOf() - leftOffset); j < subArrOut.lengthOf(); ++j)     // fill right side
//                 subArrOut.putIndexedScalar(j, subArrIn.getIndexedScalar(subArrOut.lengthOf() - j - 1));
//             break;
//         case 2:             // SYMMETRIC mode               
//             for(int j = 1;  j <= leftOffset; ++j)                                               // fill firstly left side 
//                 subArrOut.putIndexedScalar(leftOffset - j, subArrIn.getIndexedScalar(j-1));                             
//             for(int j = 0; j < subArrIn.lengthOf(); ++j)                                        // fill middle
//                 subArrOut.putIndexedScalar(leftOffset + j, subArrIn.getIndexedScalar(j));                   
//             for(int j = (subArrOut.lengthOf() - leftOffset); j < subArrOut.lengthOf(); ++j)     // fill right side
//                 subArrOut.putIndexedScalar(j, subArrIn.getIndexedScalar(subArrOut.lengthOf() - j));     
//             break;
//             }
	
//     shape::updateStrides(outputShapeInfo, input->ordering());    

//     return new ShapeList(outputShapeInfo);
    
// }





// }
// }