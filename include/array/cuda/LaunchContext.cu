
#include <array/LaunchContext.h>
#include <cuda.h>


namespace nd4j {
    cudaStream_t* LaunchContext::stream() {
        return _stream;
    }
}