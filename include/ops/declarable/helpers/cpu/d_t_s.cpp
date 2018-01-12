//
//
//

#include <ops/declarable/helpers/d_t_s.h>

namespace nd4j {
namespace ops {
namespace helpers {
    template <typename T>
    void _depthToSpace(NDArray<T> *input, NDArray<T> *output, int block_size, bool isNHWC) {

    }

    template void _depthToSpace<float>(NDArray<float> *input, NDArray<float> *output, int block_size, bool isNHWC);
    template void _depthToSpace<float16>(NDArray<float16> *input, NDArray<float16> *output, int block_size, bool isNHWC);
    template void _depthToSpace<double>(NDArray<double> *input, NDArray<double> *output, int block_size, bool isNHWC);
}
}
}