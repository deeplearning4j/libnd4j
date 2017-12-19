//
// Created by Yurii Shyrma on 18.12.2017
//


#include <DataTypeUtils.h>
#include <ops.h>
#include<ops/declarable/helpers/householder.h>

namespace nd4j {
namespace ops {
namespace helpers {


	/**
    *  this function evaluates data (coeff, normX, tail) used in Householder transformation
    *  formula for Householder matrix: P = identity_matrix - coeff * w * w^T
    *  P * x = [normX, 0, 0 , 0, ...]    
    *  w = [1, w1, w2, w3, ...], "tail" is w except first unity element, that is "tail" = [w1, w2, w3, ...]
    *  w = u / u0, vector
    *  u = x - |x|*e0, vector
    *  u0 = x0 - |x|, scalar     
    *  e0 = [±1, 0, 0, 0, ...], unit basis vector
    *  coeff = u0 / |x|, scalar    
    * 
    *  x - input vector, remains unaffected
    *  tail - output vector with length = x.lengthOf() - 1 and contains all elements of w vector except first one 
    *  normX - this scalar would be the first non-zero element in result of Householder transformation -> P*x  
    *  coeff - scalar, scaling factor in Householder matrix formula  
    */                  	

//////////////////////////////////////////////////////////////////////////
template <typename T>
void evalHouseholderData(const NDArray<T>& x, NDArray<T>& tail, T& normX, T& coeff) {

	// block of input validation
	if(!x.isVector())
		throw "ops::helpers::houseHolderForVector function: input array must be vector !";
	
	if(!tail.isVector())
		throw "ops::helpers::houseHolderForVector function: output array must be vector !";

	if(x.lengthOf() != tail.lengthOf() + 1)
		throw "ops::helpers::houseHolderForVector function: output vector must have length smaller by unity compared to input vector !";
		
	normX = x.template reduceNumber<simdOps::Norm2<T>>();	
	const T min = DataTypeUtils::min<T>();

	if(normX*normX - x(0)*x(0) <= min) {

		coeff = 0.;
		normX = x(0); 
		tail = 0.;
	}
	else {
		
		const int sign = x(0) > 0. ? -1 : +1; 					// choose opposite sign to lessen roundoff error
		const T u0 = x(0) - sign*normX;
		coeff = u0 / normX;
		if(x.isRowVector())
			tail = x({{}, {1,-1}}) / u0;
		else
			tail = x({{1,-1}, {}}) / u0;
	} 
}


template void evalHouseholderData<float>  (const NDArray<float  >& x, NDArray<float  >& tail, float  & normX, float  & coeff);
template void evalHouseholderData<float16>(const NDArray<float16>& x, NDArray<float16>& tail, float16& normX, float16& coeff);
template void evalHouseholderData<double> (const NDArray<double >& x, NDArray<double >& tail, double & normX, double & coeff);


}
}
}


