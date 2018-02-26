//
//  @author sgazeos@gmail.com
//
#ifndef __LUP_H_HELPERS__
#define __LUP_H_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    T lup(NDArray<T>* input, NDArray<T>* compound, NDArray<T>* permutation);
}
}
}
#endif
