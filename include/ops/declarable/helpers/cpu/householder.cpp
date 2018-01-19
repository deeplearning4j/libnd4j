//
// Created by Yurii Shyrma on 18.12.2017
//

#include <ops/declarable/helpers/householder.h>

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T> Householder<T>::evalHHmatrix(const NDArray<T>& x) {

	// input validation
	if(!x.isVector() && !x.isScalar())
		throw "ops::helpers::Householder::evalHHmatrix method: input array must be vector or scalar!";

	NDArray<T> w((int)x.lengthOf(), 1,  x.ordering(), x.getWorkspace());							// column-vector
	NDArray<T> wT(1, (int)x.lengthOf(), x.ordering(), x.getWorkspace());							// row-vector (transposed w)	

	T coeff;
	T normX = x.template reduceNumber<simdOps::Norm2<T>>();	
	const T min = DataTypeUtils::min<T>();
	
	if(normX*normX - x(0)*x(0) <= min) {

		normX = x(0); 
		coeff = (T)0.;		
		w = (T)0.;
		
	} 	
	else {
		
		if(x(0) >= (T)0.)
			normX = -normX;									// choose opposite sign to lessen roundoff error
		
		T u0 = x(0) - normX;
		coeff = -u0 / normX;				
		w.assign(x / u0);		
	}
	
	w(0) = (T)1.;
	wT.assign(&w);
	
	NDArray<T> identity((int)x.lengthOf(), (int)x.lengthOf(), x.ordering(), x.getWorkspace());					 
	identity.setIdentity();																			// identity matrix	

	return identity - mmul(w, wT) * coeff;	

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::evalHHmatrixData(const NDArray<T>& x, NDArray<T>& tail, T& coeff, T& normX) {

	// input validation
	if(!x.isVector() && !x.isScalar())
		throw "ops::helpers::Householder::evalHHmatrixData method: input array must be vector or scalar!";

	if(!x.isScalar() && x.lengthOf() != tail.lengthOf() + 1)
		throw "ops::helpers::Householder::evalHHmatrixData method: input tail vector must have length less than unity compared to input x vector!";

	normX = x.template reduceNumber<simdOps::Norm2<T>>();	
	const T min = DataTypeUtils::min<T>();	
		
	if(normX*normX - x(0)*x(0) <= min) {

		normX = x(0);
		coeff = (T)0.;		
		tail = (T)0.;		
	}
	else {
		
		if(x(0) >= (T)0.)
			normX = -normX;									// choose opposite sign to lessen roundoff error
		
		T u0 = x(0) - normX;
		coeff = -u0 / normX;				

		if(x.isRowVector())
			tail.assign(x({{}, {1, -1}}) / u0);		
		else
			tail.assign(x({{1, -1}, {}}) / u0);		
	}		
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::evalHHmatrixDataI(const NDArray<T>& x, T& coeff, T& normX) {

	int rows = (int)x.lengthOf()-1;
	int num = 1;
	
	if(rows == 0) {
		rows = 1;
		num = 0;
	}	
	
	NDArray<T> tail(rows, 1, x.ordering(), x.getWorkspace());
	evalHHmatrixData(x, tail, coeff, normX);

	if(x.isRowVector()) {
		NDArray<T>* temp = x.subarray({{}, {num, x.sizeAt(1)}});
		temp->assign(tail);
		delete temp;
	}
	else {		
		NDArray<T>* temp = x.subarray({{num, x.sizeAt(0)}, {}});
		temp->assign(tail);
		delete temp;
	}
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::mulLeft(NDArray<T>& matrix, const NDArray<T>& tail, const T coeff) {
	
	if(matrix.rankOf() != 2)
		throw "ops::helpers::Householder::mulLeft method: input array must be 2D matrix !";	

	if(matrix.sizeAt(0) == 1)   
    	matrix *= (T)1. - coeff;
  	
  	else {

  		NDArray<T> *pCol(nullptr), *pRow(nullptr);

		if(tail.isColumnVector()) {

			pCol = const_cast<NDArray<T>*>(&tail);
			pRow = tail.transpose();
		}
		else {

			pRow = const_cast<NDArray<T>*>(&tail);
			pCol = tail.transpose();
		}
    	    	
    	NDArray<T>* bottomPart =  matrix.subarray({{1, matrix.sizeAt(0)}, {}});
    	NDArray<T> temp = *bottomPart;
    	NDArray<T> vectorRow = mmul(*pRow, temp);
    	NDArray<T>* temp2 = matrix.subarray({{0,1}, {}});
    	vectorRow += *temp2;        	
    	*temp2 -= vectorRow * coeff;
    	*bottomPart -= mmul(*pCol, vectorRow) * coeff;    	

    	if(tail.isColumnVector())
    		delete pRow;
    	else
    		delete pCol;

    	delete bottomPart;
    	delete temp2;
	}
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::mulRight(NDArray<T>& matrix, const NDArray<T>& tail, const T coeff) {

	if(matrix.rankOf() != 2)
		throw "ops::helpers::Householder::mulRight method: input array must be 2D matrix !";
	
	if(matrix.sizeAt(1) == 1)   
    	matrix *= (T)1. - coeff;
  	
  	else {

  		NDArray<T> *pCol(nullptr), *pRow(nullptr);

		if(tail.isColumnVector()) {

			pCol = const_cast<NDArray<T>*>(&tail);
			pRow = tail.transpose();
		}
		else {

			pRow = const_cast<NDArray<T>*>(&tail);
			pCol = tail.transpose();
		}	
    	    	
		NDArray<T>* rightPart =  matrix.subarray({{}, {1, matrix.sizeAt(1)}});
    	NDArray<T> temp = *rightPart;
    	NDArray<T> vectorCol  = mmul(temp, *pCol);      	      	
    	NDArray<T>* temp2 = matrix.subarray({{},{0,1}});
    	vectorCol += *temp2;
    	*temp2 -= vectorCol * coeff;    	
    	*rightPart -= mmul(vectorCol, *pRow) * coeff;    	
    
   		if(tail.isColumnVector())
    		delete pRow;
    	else
    		delete pCol;
    	
    	delete rightPart;    	
    	delete temp2;
	}
}

      
template class ND4J_EXPORT Householder<float>;
template class ND4J_EXPORT Householder<float16>;
template class ND4J_EXPORT Householder<double>;







}
}
}
