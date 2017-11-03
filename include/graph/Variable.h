//
// @author raver119@gmail.com
//

#ifndef LIBND4J_VARIABLE_H
#define LIBND4J_VARIABLE_H

#include <string>
#include <NDArray.h>
#include <graph/generated/array_generated.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>

namespace nd4j {
    namespace graph {
        template <typename T>
        class Variable {
        protected:
            int32_t _id = 0;
            int _index = 0;
            nd4j::NDArray<T> * _ndarray = nullptr;
            std::string _name;

            bool _external = false;
            bool _readOnly = false;
            bool _placeholder = false;
            bool _removable = true;

            // for now we're setting default to numeric
            // in future we'll be fetching it right from the array, 
            InputType _variableType = InputType_UNDEFINED;
            DataType _dataType = DataType_INHERIT;

        public:
            Variable(bool placeHolder);
            Variable(nd4j::NDArray<T> *arrayw, const char *name, int id, int idx = 0);
            Variable(nd4j::NDArray<T> *array = nullptr, const char *name = nullptr);
            Variable(const nd4j::graph::FlatVariable *flatVariable);
            ~Variable();

            Variable<T>* clone();

            nd4j::NDArray<T> *getNDArray();
            void setNDArray(nd4j::NDArray<T> * array);
            bool isExternal();
            bool isReadOnly();
            bool isEmpty();

            bool isPlaceholder();
            bool hasNDArray();

            /**
             * This method returns InputType of this variable  
             */
            InputType variableType() {
                return _variableType;
            }

            void markExternal(bool reallyExternal);
            void markReadOnly(bool reallyReadOnly);
            void markRemovable(bool reallyRemovable);

            int32_t id();
            void setId(int32_t id);
            void setId(int id, int idx);

            std::string *getName();
            void setName(std::string *name);
        };
    }
}


#endif //LIBND4J_VARIABLE_H
