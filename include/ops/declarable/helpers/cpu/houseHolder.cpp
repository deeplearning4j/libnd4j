//
// Created by Yurii Shyrma on 18.12.2017
//


#include <DataTypeUtils.h>
#include<ops/declarable/helpers/houseHolder.h>

namespace nd4j {
namespace ops {
namespace helpers {



//////////////////////////////////////////////////////////////////////////
template <typename T>
void houseHolderForVector(const NDArray<T>& inVector, NDArray<T>& outVector, T& norm, T& coeff) {

	// block of input validation
	if(!inVector.isVector())
		throw "ops::helpers::houseHolderForVector function: input array must be vector !";
	
	if(!outVector.isVector())
		throw "ops::helpers::houseHolderForVector function: output array must be vector !";

	if(inVector.lengthOf() != outVector.lengthOf() + 1)
		throw "ops::helpers::houseHolderForVector function: output vector must have length smaller by unity compared to input vector !";
	
	norm = inVector.template reduceNumber<Norm2>();
	const T squaredNorm = norm * norm;	
	const T first = inVector(0);
	const T first2 = first * first;
	tailSquaredNorm = squaredNorm - first2;	
	const T min = min<T>();

	if(first2 <= min && tailSquaredNorm <= min) {

		coeff = 0.;
		norm = first; 
		outVector.assign(0.);
	}
	else {
		if(first >= 0.)
			norm = -norm;
		if(inVector.isRowVector())
			outVector = inVector({{}, {1,-1}}) / (first - norm);
		else
			outVector = inVector({{1,-1}, {}}) / (first - norm);

		coeff = (norm - first) / norm;
	}

}


// template float   zeta<float>  (const float   x, const float   q);
// template float16 zeta<float16>(const float16 x, const float16 q);
// template double  zeta<double> (const double  x, const double  q);

// template NDArray<float>   zeta<float>  (const NDArray<float>&   x, const NDArray<float>&   q);
// template NDArray<float16> zeta<float16>(const NDArray<float16>& x, const NDArray<float16>& q);
// template NDArray<double>  zeta<double> (const NDArray<double>&  x, const NDArray<double>&  q);


}
}
}

