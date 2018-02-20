//
//  @author raver119@gmail.com
//

#include <graph/profiling/GraphProfile.h>
#include <helpers/logger.h>
#include <chrono>

namespace nd4j {
    namespace graph {
        GraphProfile::GraphProfile() {
            updateLast();
        }

        GraphProfile::~GraphProfile() {
            // releasing NodeProfile pointers
            for (auto v: _profiles)
                delete v;

            _timings.clear();
        }

        void GraphProfile::addToTotal(Nd4jIndex bytes) {
            _memoryTotal += bytes;
        }

        void GraphProfile::addToActivations(Nd4jIndex bytes) {
            _memoryActivations += bytes;
        }
        
        void GraphProfile::addToTemporary(Nd4jIndex bytes) {
            _memoryTemporary += bytes;
        }
        
        void GraphProfile::addToObjects(Nd4jIndex bytes) {
            _memoryObjects += bytes;
        }

        void GraphProfile::setBuildTime(Nd4jIndex nanos) {
            _buildTime = nanos;
        }

        void GraphProfile::setExecutionTime(Nd4jIndex nanos) {
            _executionTime = nanos;
        }


        Nd4jIndex GraphProfile::currentTime() {
            auto t = std::chrono::system_clock::now();
            auto v = std::chrono::time_point_cast<std::chrono::microseconds> (t);
            auto epoch = v.time_since_epoch();
            return (Nd4jIndex) std::chrono::duration_cast<std::chrono::microseconds>(epoch).count();
        }
        
        Nd4jIndex GraphProfile::relativeTime(Nd4jIndex time) {
            auto t1 = currentTime();
            return t1 - time;
        }

        void GraphProfile::updateLast() {
            _last = std::chrono::system_clock::now();
        }

        void GraphProfile::startEvent(const char *name) {

        }

        void GraphProfile::recordEvent(const char *name) {

        }
        
        void GraphProfile::deleteEvent(const char *name) {

        }
            
        void GraphProfile::spotEvent(const char *name) {
            auto t = std::chrono::system_clock::now();
            auto d = (Nd4jIndex) std::chrono::duration_cast<std::chrono::microseconds>(t - _last).count();
            std::string k = name;
            _timings[k] = d;
            updateLast();
        }

        void GraphProfile::printOut() {
            nd4j_printf("Graph report:\n", "");
            nd4j_printf("\nMemory:\n", "");

            nd4j_printf("\nTime:\n", "");
            nd4j_printf("Construction time: %lld\n", _buildTime);
            nd4j_printf("Execution time: %lld\n", _executionTime);

            nd4j_printf("\nPer-node reports:\n", "");
            for (auto v: _profiles)
                v->printOut();
            
            nd4j_printf("\nTimers:\n", "");
            for (auto v: _timings)
                nd4j_printf("%s: %lld\n", v.first.c_str(), v.second);
        }
    }
}