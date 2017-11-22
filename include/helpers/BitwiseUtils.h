//
// Created by raver119 on 10.11.2017.
//

#ifndef LIBND4J_BITWISEUTILS_H
#define LIBND4J_BITWISEUTILS_H

#include <vector>
#include <array/ByteOrder.h>

namespace nd4j {
    class BitwiseUtils {
    public:


        /**
         * This method returns first non-zero bit index
         * @param holder
         * @return
         */
        static int valueBit(int holder);

        /**
         *  This method returns vector representation of bits.
         * 
         *  PLEASE NOTE: Result is ALWAYS left-to-right 
         */
        static std::vector<int> valueBits(int holder);

        /**
         *  This method returns TRUE if it's called on Big-Endian system, and false otherwise
         */
        static bool isBE();

        /**
         * This method returns enum
         * @return
         */
        static nd4j::ByteOrder asByteOrder();
    };
}


#endif //LIBND4J_BITWISEUTILS_H
