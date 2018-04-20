//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//


#include<ops/declarable/helpers/transforms.h>
#include <array/ResultSet.h>
#include <helpers/ShapeUtils.h>
#include <NDArrayFactory.h>
#include <numeric>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {



//////////////////////////////////////////////////////////////////////////
template <typename T>
void triu(const NDArray<T>& input, NDArray<T>& output, const int diagonal) {

    const int rank = input.rankOf();
    
    switch(rank) {

        case 1:
            for(int i = 0; i < output.sizeAt(0); ++i)
                output({{i, i+1}, {}}).assign(input);
            output.setValueInDiagMatrix(0., diagonal-1, 'l');    
            break;

        case 2:
            output.assign(input);
            output.setValueInDiagMatrix(0., diagonal-1, 'l');    
            break;

        default:            
            ResultSet<T>* inTads  = NDArrayFactory<T>::allTensorsAlongDimension(&input,  {rank-2, rank-1});
            ResultSet<T>* outTads = NDArrayFactory<T>::allTensorsAlongDimension(&output, {rank-2, rank-1});                        
#pragma omp parallel for if(inTads->size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)     
            for(int i = 0; i < inTads->size(); ++i) {
                outTads->at(i)->assign(inTads->at(i));
                outTads->at(i)->setValueInDiagMatrix(0., diagonal-1, 'l');    
            }
            delete inTads;
            delete outTads;
    }
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void triuBP(const NDArray<T>& input, const NDArray<T>& gradO, NDArray<T>& gradI, const int diagonal) {

    NDArray<T> dOdI(&gradO);                // dO/dI
    helpers::triu(input, dOdI, diagonal);

#pragma omp parallel for if(dOdI.lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)     
    for(int i = 0; i < dOdI.lengthOf(); ++i) {
        T* currElement = &dOdI(i);
        if(*currElement != (T)0.)
            *currElement = 1.;
    }

    gradI.assign(dOdI * gradO);                          // chain rule: dLoss/dI = dO/dI * dLoss/dO 
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void trace(const NDArray<T>& input, NDArray<T>& output) {

    const int inRank = input.rankOf();

    ResultSet<T>* setOfSubArrs = NDArrayFactory<T>::allTensorsAlongDimension(&input, {inRank-2, inRank-1});

#pragma omp parallel for if(setOfSubArrs->size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)     
    for(int i = 0; i < setOfSubArrs->size(); ++i)
        output(i) = setOfSubArrs->at(i)->getTrace();

    delete setOfSubArrs;
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void randomShuffle(NDArray<T>& input, NDArray<T>& output, nd4j::random::RandomBuffer& rng, const bool isInplace) {

    // check edge cases first
    int temp;
    const int firstDim = input.sizeAt(0);    
    if(input.lengthOf() == 1 || firstDim == 1) {
        
        if(!isInplace)
            output.assign(input);
    } 
    else if (input.isVector() || shape::isLikeVector(input.getShapeInfo(), temp)) {
                
        // apply Fisher-Yates shuffle 
        if(isInplace) {
#pragma omp parallel for if((firstDim-1) > Environment::getInstance()->elementwiseThreshold()) schedule(guided)       
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng.nextInt(0, i);
                if(i == r)
                    continue;
                math::nd4j_swap<T>(input(i), input(r));            
            }        
        }
        else {        
            std::vector<int> indeces(firstDim);        
            std::iota(indeces.begin(), indeces.end(), 0);        
            output(0) = input(0);
#pragma omp parallel for if((firstDim-1) > Environment::getInstance()->elementwiseThreshold()) schedule(guided)       
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng.nextInt(0, i);
                output(i) = input(indeces[r]);
                if(i == r)
                    continue;
                output(r) = input(indeces[i]);                
                math::nd4j_swap<int>(indeces[i], indeces[r]);
            }           
            rng.rewindH(firstDim-1);
        }
    }
    else {
            
        // evaluate sub-arrays list of input array through all dimensions excluding first one
        std::vector<int> dimensions = ShapeUtils<T>::evalDimsToExclude(input.rankOf(), {0});       
        ResultSet<T>* subArrsListIn = NDArrayFactory<T>::allTensorsAlongDimension(&input, dimensions);

        // apply Fisher-Yates shuffle
        if(isInplace) {
#pragma omp parallel for if((firstDim-1) > Environment::getInstance()->elementwiseThreshold()) schedule(guided)        
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng.nextInt(0, i);
                if(i == r)
                    continue;
                subArrsListIn->at(i)->swapUnsafe(*subArrsListIn->at(r));
            }        
        }
        else {
            // evaluate sub-arrays list of output array through all dimensions excluding first one        
            ResultSet<T>* subArrsListOut = NDArrayFactory<T>::allTensorsAlongDimension(&output, dimensions);        
            std::vector<int> indeces(firstDim);        
            std::iota(indeces.begin(), indeces.end(), 0);        
            bool isZeroShuffled = false;
#pragma omp parallel for if((firstDim-1) > Environment::getInstance()->elementwiseThreshold()) schedule(guided)       
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng.nextInt(0, i);
                subArrsListOut->at(i)->assign(subArrsListIn->at(indeces[r]));
                if(r == 0)
                    isZeroShuffled = true;
                if(i == r)
                    continue;
                subArrsListOut->at(r)->assign(subArrsListIn->at(indeces[i]));
                math::nd4j_swap<int>(indeces[i], indeces[r]);
            }           
            if(!isZeroShuffled)
                subArrsListOut->at(0)->assign(subArrsListIn->at(0));
            delete subArrsListOut;
        }
        rng.rewindH(firstDim-1);
        delete subArrsListIn;
    }

}

template void triu<float>(const NDArray<float>& input, NDArray<float>& output, const int diagonal);
template void triu<float16>(const NDArray<float16>& input, NDArray<float16>& output, const int diagonal);
template void triu<double>(const NDArray<double>& input, NDArray<double>& output, const int diagonal);

template void triuBP<float>(const NDArray<float>& input, const NDArray<float>& gradO, NDArray<float>& gradI, const int diagonal);
template void triuBP<float16>(const NDArray<float16>& input, const NDArray<float16>& gradO, NDArray<float16>& gradI, const int diagonal);
template void triuBP<double>(const NDArray<double>& input, const NDArray<double>& gradO, NDArray<double>& gradI, const int diagonal);

template void trace<float>(const NDArray<float>& input, NDArray<float>& output);
template void trace<float16>(const NDArray<float16>& input, NDArray<float16>& output);
template void trace<double>(const NDArray<double>& input, NDArray<double>& output);

template void randomShuffle<float>(NDArray<float>& input, NDArray<float>& output, nd4j::random::RandomBuffer& rng, const bool isInplace);
template void randomShuffle<float16>(NDArray<float16>& input, NDArray<float16>& output, nd4j::random::RandomBuffer& rng, const bool isInplace);
template void randomShuffle<double>(NDArray<double>& input, NDArray<double>& output, nd4j::random::RandomBuffer& rng, const bool isInplace);

}
}
}

