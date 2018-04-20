//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//

#ifndef LIBND4J_TRANSFORMS_H
#define LIBND4J_TRANSFORMS_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


	template <typename T>
	void triu(const NDArray<T>& input, NDArray<T>& output, const int diagonal);

	template <typename T>
	void triuBP(const NDArray<T>& input, const NDArray<T>& gradO, NDArray<T>& gradI, const int diagonal);

	template <typename T>
	void trace(const NDArray<T>& input, NDArray<T>& output);

	template <typename T>
	void randomShuffle(NDArray<T>& input, NDArray<T>& output, nd4j::random::RandomBuffer& rng, const bool isInplace);
    

}
}
}


#endif //LIBND4J_TRANSFORMS_H
