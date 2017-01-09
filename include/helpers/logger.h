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
        __host__
#endif
        static void info(const char *format, ...) {
            va_list args;
            va_start(args, format);

            vprintf(format, args);

            va_end(args);

            fflush(stdout);
        }
    };
}


#endif //LIBND4J_LOGGER_H
