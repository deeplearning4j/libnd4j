//
// @author Yurii Shyrma, created on 27.03.2018
//

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/lstmCell.h>

// namespace nd4j {
// namespace ops  {


// //////////////////////////////////////////////////////////////////////////
// CUSTOM_OP_IMPL(static_bidirectional_rnn, 7, 4, false, 0, 0) {

//     NDArray<T>* x  	  = INPUT_VARIABLE(0);                  // input [time x bS x inSize]
// 	NDArray<T>* WxFW  = INPUT_VARIABLE(1);                  // input-to-hidden  weights for forward  RNN, [inSize  x numUnits] 	
//     NDArray<T>* WxBW  = INPUT_VARIABLE(2);                  // input-to-hidden  weights for backward RNN, [inSize  x numUnits] 	
//     NDArray<T>* WhFW  = INPUT_VARIABLE(3);                  // hidden-to-hidden weights for forward  RNN, [numUnits x numUnits]         
//     NDArray<T>* WhBW  = INPUT_VARIABLE(4);                  // hidden-to-hidden weights for backward RNN, [numUnits x numUnits]         
// 	NDArray<T>* bFW   = INPUT_VARIABLE(5);                  // biases for forward  RNN, [2*numUnits] 
// 	NDArray<T>* bBW   = INPUT_VARIABLE(6);                  // biases for backward RNN, [2*numUnits] 

// 	NDArray<T>* h0FW = nullptr;								// initial cell output for forward  RNN (at time step = 0) [bS x numUnits]
// 	NDArray<T>* h0BW = nullptr;								// initial cell output for backward RNN (at time step = 0) [bS x numUnits]
// 	NDArray<T>* maxTimeStep = nullptr;						// vector [bS] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep

// 	switch(block.width()) {
// 		case 8:
// 			maxTimeStep = INPUT_VARIABLE(7);
// 			break;
// 		case 9:
// 			h0FW = INPUT_VARIABLE(7);
// 			h0FB = INPUT_VARIABLE(8);
// 			break;
// 		case 10:
// 			h0FW = INPUT_VARIABLE(7);
// 			h0FB = INPUT_VARIABLE(8);
// 			maxTimeStep = INPUT_VARIABLE(8);
// 			break;
// 	}
    
//     NDArray<T>* hFW      =  OUTPUT_VARIABLE(0);                 // cell outputs for forward  RNN [time x bS x numUnits], that is per each time step
//     NDArray<T>* hBW      =  OUTPUT_VARIABLE(1);                 // cell outputs for backward RNN [time x bS x numUnits], that is per each time step
//     NDArray<T>* hFWFinal =  OUTPUT_VARIABLE(2);                 // final cell out for forward  RNN [bS x numUnits], that is at last time step
//     NDArray<T>* hBWFinal =  OUTPUT_VARIABLE(3);                 // final cell out for backward RNN [bS x numUnits], that is at last time step

//     const int time = x->sizeAt(0);
//     const int bS   = x->sizeAt(1);
//     const int numUnits = WxFW->sizeAt(1);

//     bool ish0Allocated = false;
//     if(h0FW == nullptr) {
//     	h0FW = new NDArray<T>(x->ordering(), {bS, numUnits}, block.getWorkspace());
//     	h0BW = new NDArray<T>(x->ordering(), {bS, numUnits}, block.getWorkspace());
//     	*h0FW = 0.;
//     	*h0BW = 0.;
//     }

//     NDArray<T> curH0FW(*h0FW);
//     NDArray<T> curH0FW(*h0BW);

//     ResultSet<T>* xSubArrs = NDArrayFactory<T>::allExamples(x);
//     ResultSet<T>* hSubArrs = NDArrayFactory<T>::allExamples(h);
//     ResultSet<T>* cSubArrs = NDArrayFactory<T>::allExamples(c);

//     // forward RNN, loop through time steps
//     for (int t = 0; t < time; ++t) {

//         helpers::rnnCell<T>({&xt, &ht_1, &Wx, &Wh, &b}, &ht);
//         helpers::rnnCell<T>({xSubArrs->at(t),&currentH,&currentC, Wx,Wh,Wc,Wp, b},   {hSubArrs->at(t),cSubArrs->at(t)},   {(T)peephole, (T)projection, clippingCellValue, clippingProjValue, forgetBias});
//         currentH.assign(hSubArrs->at(t));
//         currentC.assign(cSubArrs->at(t));
//     }
    
//     delete xSubArrs;
//     delete hSubArrs;
//     delete cSubArrs;

//     return Status::OK();
// }



// DECLARE_SHAPE_FN(static_bidirectional_rnn) {    

//     // evaluate output shapeInfos
//     int *hShapeInfo(nullptr), *cShapeInfo(nullptr);
//     ALLOCATE(hShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShape->at(0)), int);
//     ALLOCATE(cShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShape->at(0)), int);    
            
//     hShapeInfo[0] = cShapeInfo[0] = 3;
//     hShapeInfo[1] = cShapeInfo[1] = inputShape->at(0)[1];
//     hShapeInfo[2] = cShapeInfo[2] = inputShape->at(0)[2];
//     hShapeInfo[3] = inputShape->at(1)[2];
//     cShapeInfo[3] = inputShape->at(2)[2];    
    
//     shape::updateStrides(hShapeInfo, shape::order(inputShape->at(1)));    
//     shape::updateStrides(cShapeInfo, shape::order(inputShape->at(2)));
         
//     return SHAPELIST(hShapeInfo, cShapeInfo);
// }   








// }
// }

//  for time, input_ in enumerate(inputs):
//       if time > 0:
//         varscope.reuse_variables()
//       # pylint: disable=cell-var-from-loop
//       call_cell = lambda: cell(input_, state)
//       # pylint: enable=cell-var-from-loop
//       if sequence_length is not None:
//         (output, state) = _rnn_step(
//             time=time,
//             sequence_length=sequence_length,
//             min_sequence_length=min_sequence_length,
//             max_sequence_length=max_sequence_length,
//             zero_output=zero_output,
//             state=state,
//             call_cell=call_cell,
//             state_size=cell.state_size)
//       else:
//         (output, state) = call_cell()

//       outputs.append(output)

//     return (outputs, state)

