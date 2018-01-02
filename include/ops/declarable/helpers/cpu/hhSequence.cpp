//
// Created by Yurii Shyrma on 02.01.2018
//

#include <ops/declarable/helpers/hhSequence.h>
#include <ops/declarable/helpers/householder.h>

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
HHsequence<T>::HHsequence(const NDArray<T>& vectors, const NDArray<T>& coeffs, const char type): _vectors(vectors), _coeffs(coeffs) {
	
	_length = nd4j::math::nd4j_min(_vectors.sizeAt(0), _vectors.sizeAt(1));
	_shift = 0;    
	_type  = type;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void HHsequence<T>::mulLeft(NDArray<T>& matrix) const {    		

	const int rows   = _vectors.sizeAt(0);
	const int cols   = _vectors.sizeAt(1);
	const int inRows = matrix.sizeAt(0);	

	NDArray<T>* block(nullptr);

	for(int i = _length - 1; i >= 0; --i) {		
    	
    	if(_type == 'u') {
    		
    		block = matrix.subarray({{inRows - rows + _shift + i, inRows}, {}});
    		Householder<T>::mulLeft(*block, _vectors({{i + 1 + _shift, rows}, {i, i+1}}), _coeffs(i));
    	}
    	else {

    		block = matrix.subarray({{inRows - cols + _shift + i, inRows}, {}});
    		Householder<T>::mulLeft(*block, _vectors({{i, i+1}, {i + 1 + _shift, cols}}), _coeffs(i));    	
    	}

    	delete block;
    }
}



template class ND4J_EXPORT HHsequence<float>;
template class ND4J_EXPORT HHsequence<float16>;
template class ND4J_EXPORT HHsequence<double>;







}
}
}
