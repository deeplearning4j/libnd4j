//
// @author iuriish@yahoo.com
//

#ifndef LIBND4J_SHAPEUTILS_H
#define LIBND4J_SHAPEUTILS_H

#include <vector>
#include <NDArray.h>

namespace nd4j {
 
    template<typename T>
    class ShapeUtils {

    public:
       
       // evaluate shape for array resulting from tensorDot operation, also evaluate shapes and permutation dimensions for transposition of two input arrays 
       static std::vector<int> evalShapeForTensorDot(const NDArray<T>* a, const NDArray<T>* b, std::vector<int>& axesA, std::vector<int>& axesB, std::vector<int>& permutAt, std::vector<int>& permutBt, std::vector<int>& shapeAt, std::vector<int>& shapeBt);

       // evaluate resulting shape after reduce operation
       static int* evalReduceShapeInfo(const char order, std::vector<int>& dimensions, const NDArray<T>& arr);
    

    };


}

#endif //LIBND4J_SHAPEUTILS_H
