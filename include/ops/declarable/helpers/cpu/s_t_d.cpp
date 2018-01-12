//
//
//

#include <ops/declarable/helpers/s_t_d.h>

namespace nd4j {
namespace ops {
namespace helpers {
    template <typename T>
    void _spaceTodepth(NDArray<T> *input, NDArray<T> *output, int block_size, bool isNHWC) {
        
    }

    template void _spaceTodepth<float>(NDArray<float> *input, NDArray<float> *output, int block_size, bool isNHWC);
    template void _spaceTodepth<float16>(NDArray<float16> *input, NDArray<float16> *output, int block_size, bool isNHWC);
    template void _spaceTodepth<double>(NDArray<double> *input, NDArray<double> *output, int block_size, bool isNHWC);
}
}
}