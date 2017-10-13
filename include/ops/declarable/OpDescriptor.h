//
// @author raver119@gmail.com
//

#ifndef LIBND4J_OPDESCRIPTOR_H
#define LIBND4J_OPDESCRIPTOR_H

#include <string>
#include <helpers/helper_hash.h>
#include <graph/generated/node_generated.h>

namespace nd4j {
    namespace ops {

        class OpDescriptor {
        protected:
            int _opNum = 0;
            std::string _opName;
            Nd4jIndex _hash;

            int _numInputs;
            int _numOutputs;

            nd4j::graph::OpClass _opClass;

            bool _divergent;
            bool _allowsInplace;

            int _tArgs = 0;
            int _iArgs = 0;

            // field for BooleanOps
            bool _scalar = false;

        public:
            // default constructor
            OpDescriptor(int numInputs, int numOutputs, std::string opName, bool allowsInplace);

            // constructor for boolean ops
            OpDescriptor(int numInputs, std::string opName, bool isScalar);
            OpDescriptor(int numInputs, const char* opName, bool isScalar);

            // default constructor
            OpDescriptor(int numInputs, int numOutputs, const char *opName, bool allowsInplace);

            // constructor for configurable op
            OpDescriptor(int numInputs, int numOutputs, const char *opName, bool allowsInplace, int tArgs, int iArgs);

            // constructor for non-configurable divergent op
            OpDescriptor(int numInputs, int numOutputs, std::string opName, bool allowsInplace, bool divergent);

            // constructor for non-configurable divergent op
            OpDescriptor(int numInputs, int numOutputs, const char *opName, bool allowsInplace, bool divergent);

            // constructor for configurable divergent op
            OpDescriptor(int numInputs, int numOutputs, const char *opName, bool allowsInplace, bool divergent, int tArgs, int iArgs);

            // default destructor
            ~OpDescriptor();

            int getNumberOfTArgs();

            int getNumberOfIArgs();

            int getNumberOfInputs();

            Nd4jIndex getHash();

            int getNumberOfOutputs();

            std::string *getOpName();

            bool isDivergent();

            bool allowsInplace();

            int getOpNum();
        };
    }
}

#endif //LIBND4J_OPDESCRIPTOR_H
