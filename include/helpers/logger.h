//
// Created by raver119 on 09.01.17.
//

#ifndef LIBND4J_LOGGER_H
#define LIBND4J_LOGGER_H

#include <cstdarg>

namespace nd4j {
    class Logger {

    public:

#ifdef __CUDACC__
        __host__ __device__
#endif
        static void info(const char *format, ...) {
            va_list args;
            va_start(args, format);

            vprintf(format, args);

            va_end(args);
        }


#ifdef __CUDACC__
        /**
        * These 2 methods are special loggers for CUDA
        */
        __device__
        static void infoGrid(const char *format, ...) {
            if (threadIdx.x != 0 && blockIdx.x != 0)
                return;

            va_list args;
            va_start(args, format);

            vprintf(format, args);

            va_end(args);
        }

        __device__
        static void infoBlock(const char *format, ...) {
            if (threadIdx.x != 0 && blockIdx.x != 0)
                return;

            va_list args;
            va_start(args, format);

            vprintf(format, args);

            va_end(args);
        }
#endif
    };
}


#endif //LIBND4J_LOGGER_H
