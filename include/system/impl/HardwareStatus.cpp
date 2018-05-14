//
// Created by raver119 on 13/05/18.
//

#include <system/HardwareStatus.h>

namespace nd4j {
    const char * HardwareStatus::getDeviceName() {
        return _deviceName.c_str();
    }

    const char * HardwareStatus::getDeviceDescription() {
        return _deviceDescription.c_str();
    }

    Nd4jIndex HardwareStatus::getMemoryUsed() {
        return _memoryUsed;
    }

    Nd4jIndex HardwareStatus::getMemoryTotal() {
        return _memoryTotal;
    }

    HardwareStatus::HardwareStatus(const HardwareStatus &other) {
        _deviceName = other._deviceName;
        _deviceDescription = other._deviceDescription;
        _memoryTotal = other._memoryTotal;
        _memoryUsed = other._memoryUsed;
    }
}
