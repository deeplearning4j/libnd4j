//
// Created by raver119 on 11.10.2017.
//

#include "MemoryUtils.h"
#include <helpers/logger.h>

bool nd4j::memory::MemoryUtils::retrieveMemoryStatistics(nd4j::memory::MemoryReport &report) {
#ifdef __APPLE__
    nd4j_debug("APPLE route\n", "");
#elif _WIN32
    nd4j_debug("WIN32 route\n", "");
#else
    nd4j_debug("LINUX route\n", "");
#endif

    return false;
}
