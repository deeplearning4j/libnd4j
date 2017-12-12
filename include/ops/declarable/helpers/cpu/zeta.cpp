//
// Created by Yurii Shyrma on 12.12.2017
//


#include <DataTypeUtils.h>
#include<ops/declarable/helpers/zeta.h>

namespace nd4j {
namespace ops {
namespace helpers {

const int maxIter = 1000000;				// max number of loop iterations 

//////////////////////////////////////////////////////////////////////////
// calculate the Hurwitz zeta function
template <typename T>
T zeta(const T x, const T q) {
	
	const T precision = (T)1e-7; 									// function stops the calculation of series when next item is <= precision
		
	// if (x <= (T)1.) 
	// 	throw("zeta function: x must be > 1 !");

	// if (q <= (T)0.) 
	// 	throw("zeta function: q must be > 0 !");

	T item;
	T result = (T)0.;

// #pragma omp declare reduction (add : double,float,float16 : omp_out += omp_in) initializer(omp_priv = (T)0.)
// #pragma omp simd private(item) reduction(add:result)
	for(int i = 0; i < maxIter; ++i) {		
		
		item = math::nd4j_pow((q + i),-x);
		result += item;
		
		if(item <= precision)
			break;
	}

	return result;
}


//////////////////////////////////////////////////////////////////////////
// calculate the Hurwitz zeta function
template <typename T>
NDArray<T> zeta(const NDArray<T>& x, const NDArray<T>& q) {

	NDArray<T> result(&x, false, x.getWorkspace());

#pragma omp parallel for if(x.lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)	
	for(int i = 0; i < x.lengthOf(); ++i)
		result(i) = zeta<T>(x(i), q(i));

	return result;
}


template float   zeta<float>  (const float   x, const float   q);
template float16 zeta<float16>(const float16 x, const float16 q);
template double  zeta<double> (const double  x, const double  q);

template NDArray<float>   zeta<float>  (const NDArray<float>&   x, const NDArray<float>&   q);
template NDArray<float16> zeta<float16>(const NDArray<float16>& x, const NDArray<float16>& q);
template NDArray<double>  zeta<double> (const NDArray<double>&  x, const NDArray<double>&  q);


}
}
}

