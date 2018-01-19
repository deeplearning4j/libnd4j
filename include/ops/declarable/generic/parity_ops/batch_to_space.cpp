//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/s_t_b.h>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(batch_to_space, 3, 1, false, 0, -2) {

        return Status::OK();
    }

    DECLARE_SHAPE_FN(batch_to_space) {
        auto in = inputShape->at(0);
        auto blocks = INPUT_VARIABLE(1);
        auto crops = INPUT_VARIABLE(2);

        const int input_dims = shape::rank(in);
        const int block_dims = (int) blocks->lengthOf();

        std::vector<int> block_shape = blocks->template asVectorT<int>();
        std::vector<int> crops_shape = crops->template asVectorT<int>();

        int removed_prefix_block_dims = 0;
        for (; removed_prefix_block_dims < block_dims; ++removed_prefix_block_dims) {
            const int dim = removed_prefix_block_dims;
            if (crops_shape[2 * dim] != 0 || crops_shape[2 * dim + 1] != 0 || block_shape[dim] != 1)
                break;
        }

        int removed_suffix_block_dims = 0;
        for (; removed_suffix_block_dims < block_dims - removed_prefix_block_dims; ++removed_suffix_block_dims) {
            const int dim = block_dims - 1 - removed_suffix_block_dims;
            if (crops_shape[2 * dim] != 0 || crops_shape[2 * dim + 1] != 0 || block_shape[dim] != 1)
                break;
        }

        int block_shape_product = 1;
        for (int block_dim = 0; block_dim < block_dims; ++block_dim)
            block_shape_product *= block_shape[block_dim];


        const int orig_input_batch_size = shape::sizeAt(in, 0);
        const int internal_block_dims = block_dims - removed_prefix_block_dims - removed_suffix_block_dims;

        if (internal_block_dims == 0) {
            // just return input shape here
            int *newShape;
            COPY_SHAPE(in, newShape);
            return new ShapeList(newShape);
        }

        // go full route otherwise
        std::vector<int> internal_input_shape;
        std::vector<int> internal_output_shape;
        std::vector<int> external_output_shape;

        external_output_shape.emplace_back(orig_input_batch_size / block_shape_product);

        int input_batch_size = orig_input_batch_size;
        for (int block_dim = 0; block_dim < removed_prefix_block_dims; ++block_dim) {
            const int size = shape::sizeAt(in, block_dim + 1);
            input_batch_size *= size;
            external_output_shape.emplace_back(size);
        }
        internal_input_shape.emplace_back(input_batch_size);
        internal_output_shape.emplace_back(input_batch_size / block_shape_product);

        for (int block_dim = removed_prefix_block_dims;
             block_dim < block_dims - removed_suffix_block_dims; ++block_dim) {
            const int crop_start = crops_shape[2 * block_dim];
            const int crop_end = crops_shape[2 * block_dim + 1];

            const int input_size = shape::sizeAt(in, block_dim + 1);
            const int block_shape_value = block_shape[block_dim];
            const int cropped_size = input_size * block_shape_value - crop_start - crop_end;

            internal_input_shape.emplace_back(input_size);
            internal_output_shape.emplace_back(cropped_size);
            external_output_shape.emplace_back(cropped_size);
        }

        int depth = 1;
        for (int dim = block_dims - removed_suffix_block_dims + 1; dim < input_dims; ++dim) {
            const int size = shape::sizeAt(in, dim);
            external_output_shape.emplace_back(size);
            depth *= size;
        }

        int *newShape;
        ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength((int) external_output_shape.size()), int);

        // we always give out C order here
        shape::shapeBuffer((int) external_output_shape.size(), external_output_shape.data(), newShape);

        return new ShapeList(newShape);
    }
}
}