//
// Created by raver119 on 10.11.2017.
//

#include <helpers/logger.h>
#include <helpers/BitwiseUtils.h>

namespace nd4j {

     bool BitwiseUtils::isBE() {
        short int word = 0x0001;
        char *byte = (char *) &word;
        return(byte[0] ? false : true);
     }

    int BitwiseUtils::valueBit(int holder) {
#ifdef __LITTLE_ENDIAN__
        for (int e = 0; e < 32; e++) {
#elif __BIG_ENDIAN__
        for (int e = 32; e >= 0; e--) {
#endif
            bool isOne = (holder & 1 << e) != 0;

            if (isOne)
                return e;
        }

        return -1;
    }


    std::vector<int> BitwiseUtils::valueBits(int holder) {
        std::vector<int> bits;

#ifdef __LITTLE_ENDIAN__
        for (int e = 0; e < 32; e++) {
#elif __BIG_ENDIAN__
        for (int e = 32; e >= 0; e--) {
#endif
            bool isOne = (holder & 1 << e) != 0;

            if (isOne)
                bits.emplace_back(1);
            else
                bits.emplace_back(0);
        }

        return bits;
    }
}
