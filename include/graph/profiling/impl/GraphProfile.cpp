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

        NodeProfile* GraphProfile::nodeById(int id, const char *name) {
            if (_profilesById.count(id) == 0) {
                auto node = new NodeProfile(id, name);
                _profiles.emplace_back(node);
                _profilesById[id] = node;
                return node;
            }

            return _profilesById[id];
        }

        void GraphProfile::printOut() {
            nd4j_printf("Graph report:\n", "");
            nd4j_printf("\nMemory:\n", "");

            Nd4jIndex tmp = 0L;
            Nd4jIndex obj = 0L;
            Nd4jIndex act = 0L;
            for (auto v: _profiles) {
                tmp += v->getTemporarySize();
                obj += v->getObjectsSize();
                act += v->getActivationsSize();
            }

            nd4j_printf("ACT: %lld; TMP: %lld; OBJ: %lld; \n", act, tmp, obj);

            nd4j_printf("\nTime:\n", "");
            nd4j_printf("Construction time: %lld us;\n", _buildTime);
            nd4j_printf("Execution time: %lld us;\n", _executionTime);

            nd4j_printf("\nPer-node reports:\n", "");
            for (auto v: _profiles)
                v->printOut();
            
            nd4j_printf("\nSpecial timers:\n", "");
            for (auto v: _timings)
                nd4j_printf("%s: %lld us;\n", v.first.c_str(), v.second);
        }
    }
}