//
// @author raver119@gmail.com
//

#ifndef LIBND4J_VARIABLESPACE_H
#define LIBND4J_VARIABLESPACE_H

#include <helpers/logger.h>
#include <helpers/helper_random.h>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <mutex>
#include <NDArray.h>
#include <array/NDArrayList.h>
#include <graph/Variable.h>
#include <memory/Workspace.h>
#include <graph/Stash.h>
#include <graph/FlowPath.h>


namespace nd4j {
    namespace graph {

        template <typename T>
        class VariableSpace {
        protected:

            nd4j::memory::Workspace _workspace;
            nd4j::random::RandomBuffer* _rng = nullptr;

            // stash is NOT cloned
            nd4j::graph::Stash<T> _stash;

            std::map<std::pair<int, int>, Variable<T> *> _paired;
            std::map<std::string, Variable<T> *> _symbolic;
            std::map<int, Variable<T> *> _variables;
            std::vector<Variable<T> *> _external;
            std::vector<Variable<T> *> _internal;

            std::vector<nd4j::NDArrayList<T> *> _lists;

            std::vector<nd4j::graph::Variable<T> *> _placeholders;

            void silentPutVariable(std::pair<int,int>& pair, Variable<T> *variable);

            int _auto_counter = -1;

            std::mutex _varmap;

            std::map<int, nd4j::graph::Variable<T> *> _temporary;

            std::vector<nd4j::graph::Variable<T> *> *_handles;

            FlowPath* _flow = nullptr;

        public:
            VariableSpace();
            virtual ~VariableSpace();

            virtual VariableSpace<T>& operator=(const VariableSpace<T>& other);

            virtual int numberOfPlaceholders();
            virtual std::vector<Variable<T>*>* getPlaceholders();
            virtual nd4j::random::RandomBuffer* getRNG();
            virtual void setRNG(nd4j::random::RandomBuffer* rng);
            
            virtual nd4j::memory::Workspace *workspace();

            virtual bool hasExternalVariable(int it);
            virtual bool hasExternalVariable(std::pair<int,int>& pair);
            virtual bool hasExternalVariable(std::string *symbol);

            virtual bool hasVariable(int id);
            virtual bool hasVariable(int id, int idx);
            virtual bool hasVariable(std::pair<int,int>& pair);
            virtual bool hasVariable(std::string *symbol);

            virtual nd4j::graph::Variable<T> *getVariable(int id);
            virtual nd4j::graph::Variable<T> *getVariable(int id, int idx);
            virtual nd4j::graph::Variable<T> *getVariable(std::pair<int,int>& pair);
            virtual nd4j::graph::Variable<T> *getVariable(std::string *symbol);

            virtual void putVariable(std::pair<int,int>& pair, NDArray<T> *array);
            virtual void putVariable(std::pair<int,int>& pair, Variable<T> *variable);
            virtual void putVariable(int id, Variable<T> *variable);
            virtual void putVariable(int id, NDArray<T> *array);
            virtual void putVariable(int id, int idx, NDArray<T> *array);
            virtual void putVariable(int id, int idx, Variable<T> *array);

            virtual void dropVariable(std::pair<int,int> &pair);
            virtual void dropVariable(int id, int idx);

            virtual void trackList(nd4j::NDArrayList<T>* list);

            virtual void putOutputVariable(Variable<T> *variable);

            // memory-related statistics
            virtual Nd4jIndex externalMemory();
            virtual Nd4jIndex internalMemory();
            virtual Nd4jIndex totalMemory();

            virtual int externalEntries();
            virtual int internalEntries();
            virtual int totalEntries();

            virtual nd4j::graph::VariableSpace<T>* clone();

            template <typename N>
            nd4j::graph::VariableSpace<N>* asT();

            virtual nd4j::graph::Stash<T>* getStash();

            virtual std::vector<nd4j::graph::Variable<T> *> * getExternalVariables();

            virtual void setFlowPath(FlowPath* timers);
            virtual FlowPath* flowPath();
        };
    }
}


#endif //LIBND4J_VARIABLESPACE_H
