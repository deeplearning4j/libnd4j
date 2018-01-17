//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/parity_ops.h>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(space_to_batch, 3, 1, false, 0, -2) {
        auto input = INPUT_VARIABLE(0);
        auto blocks = INPUT_VARIABLE(1);
        auto padding = INPUT_VARIABLE(2);

        REQUIRE_TRUE(input->rankOf() >= 1 + blocks->lengthOf() + 1, 0, "SpaceToBatch: blocks length + 2 should match input rank at least");
        REQUIRE_TRUE(padding->rankOf() == 2, 0, "SpaceToBatch: padding should have rank of 2, but got %i instead", padding->rankOf());
        REQUIRE_TRUE(padding->columns() == 2 && blocks->lengthOf() == padding->rows(), 0, "SpaceToBatch: padding should have M rows and 2 columns");

        return Status::OK();
    }

    DECLARE_SHAPE_FN(space_to_batch) {
        return new ShapeList();
    }
}
}