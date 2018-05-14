//
// Created by raver119 on 13/05/18.
//

#ifndef LIBND4J_HARDWAREFEATURES_H
#define LIBND4J_HARDWAREFEATURES_H

#include <dll.h>


namespace nd4j {
    enum ND4J_EXPORT HardwareFeatures {
        NUMA = 1,
        ECC = 2,

        AVX = 4,
        AVX2 = 8,
        AVX512 = 16,

        // native fp16 support
        FP16 = 32,

        // just a stub
        LAST_FEATURE = 4294967296,
    };
}

#endif //LIBND4J_HARDWAREFEATURES_H
