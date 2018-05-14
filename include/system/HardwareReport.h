//
// Created by raver119 on 13/05/18.
//

#ifndef LIBND4J_HARDWAREREPORT_H
#define LIBND4J_HARDWAREREPORT_H

#include <system/HardwareStatus.h>
#include <pointercast.h>
#include <dll.h>
#include <vector>
#include <map>

namespace nd4j {
    class ND4J_EXPORT HardwareReport {
    private:
        // processors
        std::map<int, HardwareStatus> _processors;

        // cores
        std::map<int, HardwareStatus> _cores;

        // special devices (i.e. GPUs)
        std::map<int, HardwareStatus> _devices;
    public:
        HardwareReport() = default;
        ~HardwareReport() = default;
        HardwareReport(const HardwareReport &other);


        int getNumberOfProcessors();
        int getNumberOfCores();
        int getNumberOfDevices();

        HardwareStatus getDeviceStatus(int deviceId);
    };
}


#endif //LIBND4J_HARDWAREREPORT_H
