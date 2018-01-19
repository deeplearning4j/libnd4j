//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/s_t_b.h>

namespace nd4j {
namespace ops {
    const int kMaxSpaceToBatchBlockDims = 4;

    CUSTOM_OP_IMPL(space_to_batch, 3, 1, false, 0, -2) {
        auto input = INPUT_VARIABLE(0);
        auto blocks = INPUT_VARIABLE(1);
        auto padding = INPUT_VARIABLE(2);

        auto output = OUTPUT_VARIABLE(0);

        REQUIRE_TRUE(blocks->isVector(), 0, "SpaceToBatch: blocks supposed to be vector, but got %iD instead", blocks->rankOf());
        REQUIRE_TRUE(input->rankOf() >= 1 + blocks->lengthOf() + 1, 0, "SpaceToBatch: blocks length + 2 should match input rank at least");
        REQUIRE_TRUE(padding->rankOf() == 2, 0, "SpaceToBatch: padding should have rank of 2, but got %i instead", padding->rankOf());
        REQUIRE_TRUE(padding->columns() == 2 && blocks->lengthOf() == padding->rows(), 0, "SpaceToBatch: padding should have M rows and 2 columns");

        const int xRank = input->rankOf();
        const int block_dims = (int) blocks->lengthOf();
        std::vector<int> block_shape((int) block_dims);
        std::vector<int> padding_shape((int) padding->lengthOf());

        for (int e = 0; e < block_dims; e++)
            block_shape[e] = (int) blocks->getScalar(e);

        for (int e = 0; e < padding->lengthOf(); e++)
            padding_shape[e] = (int) padding->getScalar(e);


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

        int block_shape_product = 1;
        for (int block_dim = 0; block_dim < block_dims; ++block_dim)
            block_shape_product *= block_shape[block_dim];

        REQUIRE_TRUE(block_shape_product > 0, 0, "SpaceToBatch: block should contain values >= 1 ONLY");

        const int internal_block_dims = block_dims - removed_prefix_block_dims - removed_suffix_block_dims;

        REQUIRE_TRUE(internal_block_dims <= kMaxSpaceToBatchBlockDims, 0, "SpaceToBatch: Maximum number of non-combined block dimensions should be less or equal then %i but got %i instead", kMaxSpaceToBatchBlockDims, internal_block_dims);

        if (internal_block_dims == 0) {
            // we return array if there's nothing to move here
            output->assign(input);
            return Status::OK();
        }

        std::vector<int> internal_input_shape;
        std::vector<int> internal_output_shape;
        std::vector<int> external_output_shape;

        external_output_shape.emplace_back(input->sizeAt(0) * block_shape_product);
        int input_batch_size = input->sizeAt(0);
        for (int block_dim = 0; block_dim < removed_prefix_block_dims; block_dim++) {
            const int size = input->sizeAt(block_dim + 1);
            input_batch_size *= size;
            external_output_shape.emplace_back(size);
        }
        internal_input_shape.emplace_back(input_batch_size);
        internal_output_shape.emplace_back(input_batch_size * block_shape_product);

        for (int block_dim = removed_prefix_block_dims; block_dim < block_dims - removed_suffix_block_dims; block_dim++) {
            const int pad_start = (int) padding->getScalar(2 * block_dim);
            const int pad_end = (int) padding->getScalar(2 * block_dim + 1);

            const int input_size = input->sizeAt(block_dim + 1);
            const int block_shape_value = block_shape[block_dim];
            const int padded_size = input_size + pad_start + pad_end;
            const int output_size = padded_size / block_shape_value;

            // FIXME: validation required here

            internal_input_shape.emplace_back(input_size);
            internal_output_shape.emplace_back(output_size);
            external_output_shape.emplace_back(output_size);
        }

        int depth = 1;
        for (int dim = block_dims - removed_suffix_block_dims + 1; dim < xRank; dim++) {
            const int size = input->sizeAt(dim);
            external_output_shape.emplace_back(size);
            depth *= size;
        }

        internal_input_shape.emplace_back(depth);
        internal_output_shape.emplace_back(depth);


        helpers::_spaceToBatch(input, output, internal_input_shape, internal_output_shape, block_shape, padding_shape);


        return Status::OK();
    }

    DECLARE_SHAPE_FN(space_to_batch) {
        auto in = inputShape->at(0);
        auto blocks = INPUT_VARIABLE(1);
        auto padding = INPUT_VARIABLE(2);


        const int xRank = shape::rank(in);
        const int block_dims = (int) blocks->lengthOf();
        std::vector<int> block_shape;

        for (int e = 0; e < block_dims; e++)
            block_shape.emplace_back((int) blocks->getScalar(e));

        int removed_prefix_block_dims = 0;
        for (; removed_prefix_block_dims < block_dims; ++removed_prefix_block_dims) {
            const int dim = removed_prefix_block_dims;
            if ((int) padding->getScalar(2 * dim) != 0 || (int) padding->getScalar(2 * dim + 1) != 0 || block_shape[dim] != 1)
                break;
        }

        int removed_suffix_block_dims = 0;
        for (; removed_suffix_block_dims < block_dims - removed_prefix_block_dims; ++removed_suffix_block_dims) {
            const int dim = block_dims - 1 - removed_suffix_block_dims;
            if ((int) padding->getScalar(dim * 2) != 0 || (int) padding->getScalar(dim * 2 + 1) != 0 || block_shape[dim] != 1)
                break;
        }

        int block_shape_product = 1;
        for (int block_dim = 0; block_dim < block_dims; ++block_dim)
            block_shape_product *= block_shape[block_dim];

        const int internal_block_dims = block_dims - removed_prefix_block_dims - removed_suffix_block_dims;

        if (internal_block_dims == 0) {
            // just return input shape here
            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(in), int);
            COPY_SHAPE(in, newShape);
            return new ShapeList(newShape);   
        }

        // go full route otherwise
        std::vector<int> internal_input_shape;
        std::vector<int> internal_output_shape;
        std::vector<int> external_output_shape;

        external_output_shape.emplace_back(shape::sizeAt(in, 0) * block_shape_product);
        int input_batch_size = shape::sizeAt(in, 0);
        for (int block_dim = 0; block_dim < removed_prefix_block_dims; block_dim++) {
            const int size = shape::sizeAt(in, block_dim + 1);
            input_batch_size *= size;
            external_output_shape.emplace_back(size);
        }
        internal_input_shape.emplace_back(input_batch_size);
        internal_output_shape.emplace_back(input_batch_size * block_shape_product);

        for (int block_dim = removed_prefix_block_dims; block_dim < block_dims - removed_suffix_block_dims; block_dim++) {
            const int pad_start = (int) padding->getScalar(2 * block_dim);
            const int pad_end = (int) padding->getScalar(2 * block_dim + 1);

            const int input_size = shape::sizeAt(in, block_dim + 1);
            const int block_shape_value = block_shape[block_dim];
            const int padded_size = input_size + pad_start + pad_end;
            const int output_size = padded_size / block_shape_value;

            // FIXME: validation required here

            internal_input_shape.emplace_back(input_size);
            internal_output_shape.emplace_back(output_size);
            external_output_shape.emplace_back(output_size);
        }

        int depth = 1;
        for (int dim = block_dims - removed_suffix_block_dims + 1; dim < xRank; dim++) {
            const int size = shape::sizeAt(in, dim);
            external_output_shape.emplace_back(size);
            depth *= size;
        }

        internal_input_shape.emplace_back(depth);
        internal_output_shape.emplace_back(depth);

        int *newShape;
        ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength((int) external_output_shape.size()), int);

        // we always give out C order here
        shape::shapeBuffer((int) external_output_shape.size(), external_output_shape.data(), newShape);

        return new ShapeList(newShape);
    }
}
}