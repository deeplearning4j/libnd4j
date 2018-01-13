//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#include <dll.h>
#include <helpers/logger.h>

namespace nd4j {
    class ND4J_EXPORT Status {
    public:
        static FORCEINLINE Nd4jStatus OK() {
            return ND4J_STATUS_OK;
        };

        static FORCEINLINE Nd4jStatus Code(Nd4jStatus code, const char *message) {
            nd4j_printf("%s\n", message);
            return code;
        }
    };
}