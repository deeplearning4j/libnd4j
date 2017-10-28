//
// @author raver119@gmail.com
//

#ifndef LIBND4J_DECLARABLE_OPS_H
#define LIBND4J_DECLARABLE_OPS_H

#include <sstream>
#include <types/float16.h>
#include <pointercast.h>
#include <NDArray.h>
#include <Block.h>
#include "OpDescriptor.h"
#include <helpers/helper_hash.h>
#include <ShapeList.h>
#include <ArrayList.h>
//#include <ops/declarable/declarable_ops.h>

#include <chrono>
#include <ctime>

using namespace nd4j::graph;

namespace nd4j {
    namespace ops {

        Nd4jStatus ND4J_EXPORT conditionHelper(const char *file, int line, int condition, int argNumber, const char *format, ...);


        template<typename T>
        Nd4jStatus resultHelper(T status, const char *func, const char *file, int line) {
            if (status) {
                //  TODO: fill out error codes here
                fprintf(stderr, "Validation error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                        static_cast<unsigned int>(status), "", func);

                return ND4J_STATUS_BAD_INPUT;
            }

            return ND4J_STATUS_OK;
        }

        /**
         * This class is the basic building block of Graph Operations. Any CustomOp out there is built on top of this "abstract" class.
         *
         */
        template <typename T>
        class DeclarableOp {
        protected:
            Block<T> *_block;
            OpDescriptor *_descriptor;

            /**
             * This method executes this Op, and defined for most of individual ops separately
             */
            virtual Nd4jStatus validateAndExecute(Block<T>& block) = 0;


            /**
             * This method ensures that target variable has enough space for op execution
             *
             * TODO: we want workspaces support right here
             */
            bool allocateResult(Block<T>& block, std::initializer_list<int>& shape, char order = 'c');
            bool allocateResult(Block<T>& block, int* shape);

            /*
            * This method attaches array to specific Variable, identified by node ID and outputNumber (which is output index for multi-output operations)
            */
            void storeResult(Block<T> &block, int outputNumber, NDArray<T>& array);
            nd4j::NDArray<T> *getZ(Block<T>& block, int inputId = 0);

            /**
            *   This method pre-allocates NDArrays for Op output, in case they are not available at op execution time
            */
            bool prepareOutputs(Block<T>& block);

            //std::vector<int>* calculateOutputShape(std::vector<int>* inputShape, nd4j::graph::Block<T>& block);
        public:
            // for special cases, like BooleanOps
            DeclarableOp();
            DeclarableOp(const char *name, int numInputs, bool scalar);

            // regular constructors
            DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace);
            DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, bool divergent);
            DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, int tArgs, int iArgs);

            // for LogicalOps
            DeclarableOp(const char *name, bool isLogical);

            // default testructor
            ~DeclarableOp();

            // this method returns OpDescriptor, describing this Op instance
            OpDescriptor *getOpDescriptor();

            /**
            *   This method should be available in each implemented Op, and should return Op output shape(s), for a given input shape(s)
            */
            virtual ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Block<T>& block) = 0;

            /**
             * Returns opName
             *
             * @return
             */
            std::string *getOpName();

            /**
             * Returns opHash
             */
            Nd4jIndex getOpHash();

            /**
             * This method sets arguments for op
             */
//            void setArguments();

            /**
             * This method returns pointer to results
             */
//            void getResults();

            /**
             * This method executes given Op
             *
             * @param block
             * @return 0 if OK, error code otherwise
             */
            virtual Nd4jStatus execute(Block<T>* block);

            nd4j::ArrayList<T>* execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<T> tArgs, std::initializer_list<int> iArgs, bool isInplace = false);
            Nd4jStatus execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<NDArray<T>*> outputs , std::initializer_list<T> tArgs, std::initializer_list<int> iArgs, bool isInplace = false);

            nd4j::ArrayList<T>* execute(std::vector<NDArray<T>*>& inputs, std::vector<T>& tArgs, std::vector<int>& iArgs, bool isInplace = false);
            Nd4jStatus execute(std::vector<NDArray<T>*>& inputs, std::vector<NDArray<T>*>& outputs , std::vector<T>& tArgs, std::vector<int>& iArgs, bool isInplace = false);

            // There methods provide various validation options
            Nd4jStatus validateNonEmptyInput(Block<T>& block);

            // this method checks if all input arrays have equal lengths
            Nd4jStatus validateInputLengthMatch(Block<T>& block);

            // this method checks if all input arrays have the same shapes (orders/strides are NOT checked)
            Nd4jStatus validateInputDimensionsMatch(Block<T>& block);

            // this method check if all input arrays have the same orders
            Nd4jStatus validateOrdersMatch(Block<T>& block);

            // this method checks if all input arrays are 2D
            Nd4jStatus validateInput2D(Block<T>& block);

            // this method checks if all input arrays are 3D
            Nd4jStatus validateInput3D(Block<T>& block);

            // this method checks if all input arrays are 4D
            Nd4jStatus validateInput4D(Block<T>& block);

            // this method checks if all input arrays are ND
            Nd4jStatus validateInputDimensions(Block<T>& block, int rank);

            // this method checks if number of available arguments matches op expectations
            Nd4jStatus validateArguments(Block<T>& block);
        };
    }
}

#endif //LIBND4J_DECLARABLE_OPS_H
