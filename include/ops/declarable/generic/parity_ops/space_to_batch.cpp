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

        REQUIRE_TRUE(blocks->isVector(), 0, "SpaceToBatch: blocks supposed to be vector, but got %iD instead", blocks->rankOf());
        REQUIRE_TRUE(input->rankOf() >= 1 + blocks->lengthOf() + 1, 0, "SpaceToBatch: blocks length + 2 should match input rank at least");
        REQUIRE_TRUE(padding->rankOf() == 2, 0, "SpaceToBatch: padding should have rank of 2, but got %i instead", padding->rankOf());
        REQUIRE_TRUE(padding->columns() == 2 && blocks->lengthOf() == padding->rows(), 0, "SpaceToBatch: padding should have M rows and 2 columns");

        const int xRank = input->rankOf();
        const int block_dims = blocks->lengthOf();
        std::vector<int> block_shape;

        for (int e = 0; e < block_dims; e++)
            block_shape.emplace_back((int) blocks->getScalar(e));


        // Determine the length of the prefix of block dims that can be combined
        // into the batch dimension due to having no padding and block_shape=1.
        int removed_prefix_block_dims = 0;
        for (; removed_prefix_block_dims < block_dims; ++removed_prefix_block_dims) {
            const int dim = removed_prefix_block_dims;
            if ((int) padding->getScalar(2 * dim) != 0 || (int) padding->getScalar(2 * dim + 1) != 0 || block_shape[dim] != 1)
                break;            
        }

        // Determine the length of the suffix of block dims that can be combined
        // into the depth dimension due to having no padding and block_shape=1.
        int removed_suffix_block_dims = 0;
        for (; removed_suffix_block_dims < block_dims - removed_prefix_block_dims; ++removed_suffix_block_dims) {
            const int dim = block_dims - 1 - removed_suffix_block_dims;
            if ((int) padding->getScalar(dim * 2) != 0 || (int) padding->getScalar(dim * 2 + 1) != 0 || block_shape[dim] != 1)
                break;
        }


        return Status::OK();
    }

    DECLARE_SHAPE_FN(space_to_batch) {
        return new ShapeList();
    }
}
}