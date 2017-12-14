//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation returns index of max element in a given NDArray (optionally: along given dimension(s))
         * Expected input:
         * 0: N-dimensional array
         * 1: optional axis vector
         * 
         * Int args:
         * 0: optional axis
         */
        DECLARE_REDUCTION_OP(argmax, 1, 1, false, 0, -2);

        /**
         * This operation returns index of min element in a given NDArray (optionally: along given dimension(s))
         * Expected input:
         * 0: N-dimensional array
         * 1: optional axis vector
         * 
         * Int args:
         * 0: optional axis
         */
        DECLARE_REDUCTION_OP(argmin, 1, 1, false, 0, -2);

        /**
         * This operation provides various normalization modes: 
         * 0: frobenius
         * 1: euclidean (norm2)
         * 2: norm1
         * 3: norm2
         * 4: inf-norm
         * 5: p-norm
         * 
         * Expected arguments:
         * input: N-dimensional array
         * 
         * 
         * Int args:
         * 0...: axis
         * 
         * T args:
         * 0: norm mode
         * 1: p for p-norm
         */
        DECLARE_REDUCTION_OP(norm, 1, 1, false, 1, -2);
           
        /**
         * Returns a batched matrix tensor with new batched diagonal values.
         */
        DECLARE_CONFIGURABLE_OP(matrix_set_diag, 2, 1, false, 0, 0);


        DECLARE_CONFIGURABLE_OP(betainc, 3, 1, false, 0, 0);

        /**
         * This operation is added for compatibility purposes mostly.
         * PLEASE NOTE: Please consider using Add instead
         * Expected arguments:
         * 0: N-dimensional input
         * 1: bias vector
         */
        DECLARE_OP(biasadd, 2, 1, true);
        DECLARE_CUSTOM_OP(biasadd_bp, 3, 2, false, 0, 0);

        /**
         * Returns a diagonal tensor with a given diagonal values. Given a diagonal, this operation returns a tensor with the diagonal and everything else padded with zeros.
         */
        DECLARE_CUSTOM_OP(diag, 1, 1, false, 0, 0);

        /**
         * Returns a diagonal tensor with a given diagonal values. Given a diagonal, this operation returns a tensor with the diagonal and everything else padded with zeros.
         */
        DECLARE_CUSTOM_OP(diag_part, 1, 1, false, 0, 0);

        /**
         * This operation takes 2 arrays: original values, and values to be excluded. And returns 2 arrays: values left after exclusion, and indices in original array for surivals.
         * Expected arguments:
         * 0: vector with original values
         * 1: vector with values to exclude
         */
        DECLARE_OP(listdiff, 2, 2, false);

        /**
         * This operation applies Add opeartion to specific inputs wrt indices
         * Expected arguments:
         * input: N-dimensional array
         * indices: either scalar, vector, or N-dimensional array
         * updates: N-dimensional array
         */
        DECLARE_OP(scatter_add, 3, 1, true);

        /**
         * This operation applies Subtract opeartion to specific inputs wrt indices
         * Expected arguments:
         * input: N-dimensional array
         * indices: either scalar, vector, or N-dimensional array
         * updates: N-dimensional array
         */
        DECLARE_OP(scatter_sub, 3, 1, true);

        /**
         * This operation applies Multiply opeartion to specific inputs wrt indices
         * Expected arguments:
         * input: N-dimensional array
         * indices: either scalar, vector, or N-dimensional array
         * updates: N-dimensional array
         */
        DECLARE_OP(scatter_mul, 3, 1, true);

        /**
         * This operation applies Divide opeartion to specific inputs wrt indices
         * Expected arguments:
         * input: N-dimensional array
         * indices: either scalar, vector, or N-dimensional array
         * updates: N-dimensional array
         */
        DECLARE_OP(scatter_div, 3, 1, true);

        /**
         * This operation applies Assign opeartion to specific inputs wrt indices
         * Expected arguments:
         * input: N-dimensional array
         * indices: either scalar, vector, or N-dimensional array
         * updates: N-dimensional array
         */
        DECLARE_OP(scatter_upd, 3, 1, true);

        /**
         * This operation takes input's shape, and returns new NDArray filled with specified value
         * Expected arguments:
         * input: N-dimensional array
         * 
         * T args:
         * 0: scalar value, used to fill NDArray
         */
        DECLARE_CONFIGURABLE_OP(fill_as, 1, 1, true, 1, 0);

        /**
         * This operation applies element-wise rint (round to integral value) operation
         */
        DECLARE_OP(rint, 1, 1, true);

        /**
         * This operation returns unique elements from input array as vector, and their original indices in input array
         * Expected input:
         * input: N-dimensional array
         */
        DECLARE_CUSTOM_OP(unique, 1, 2, false, 0, 0);

        /**
         * This operation splits input NDArray into multiple TADs along given dimensions
         * Expected arguments:
         * input: N-dimensional array
         * 
         * Int args:
         * 0..: TAD axis
         */
        DECLARE_CUSTOM_OP(tear, 1, -1, false, 0, -1);

        /**
         * This op does the same as tear, just uses different input format:
         * @tparam T
         */
        DECLARE_CUSTOM_OP(unstack, 1, -1, false, 0, 1);

        /**
         * This operation extracts a strided (optionally) slice from a tensor, 
         */
        DECLARE_CUSTOM_OP(strided_slice, 1, 1, false, 0, 5); // TODO: new op type needed. that returns VIEW

        /**
         * This operation extracts a slice from a tensor.
         * 
         */
        DECLARE_CUSTOM_OP(slice, 1, 1, false, 0, -1);

        /**
         * This operation generate sequences. Basically from......to, with step used as increment.
         * Expected arguments:
         * start: optional scalar with starting value
         * stop: optional scalar with end value
         * step: optional scalar witn step value
         * 
         * Int args: (optional)
         * 0: optional scalar with starting value
         * 1: optional scalar with end value
         * 1: optional scalar witn step value
         * 
         * T args: (optional)
         * 0: optional scalar with starting value
         * 1: optional scalar with end value
         * 1: optional scalar witn step value
         */
        DECLARE_CUSTOM_OP(range, -2, 1, false, -2, -2);

        /**
         * This operation return one-hot encoded n-dimensional array
         * Expected arguments:
         * input: N-dimensional array
         * 
         * T args:
         * 0: 'on' value
         * 1: 'off' value
         * 
         * Int args:
         * 0: depth
         * 1: axis
         */
        DECLARE_CUSTOM_OP(onehot, 1, 1, false, 2, 2);

        /**
		 * This operation stacks a list of rank tensors into one rank-(R+1) tensor.
		 * Expected arguments:
		 * 0...: N-Dimensional arrays to stack
		 * 
		 */
        DECLARE_CUSTOM_OP(stack, -1, 1, false, 0, 0);

        /**
         * This operation returns length of input array
         * Expected arguments:
         * input: N-dimensional array
         * 
         * TODO: make this operation reduction, to allow TAD -> size
         */
        DECLARE_CUSTOM_OP(size, 1, 1, false, 0, 0); // add DeclarableScalarOp?


        /**
         * This operation returns rank of input array as scalar value.
         */
        DECLARE_CUSTOM_OP(rank, 1, 1, false, 0, 0); // ^


        DECLARE_OP(broadcastgradientargs, 2, 2, true);

        /**
         * This operation takes input's shape, and returns new NDArray filled with zeros
         * Expected arguments:
         * input: N-dimensional array
         * 
         */
        DECLARE_OP(zeros_as, 1, 1, false);

        /**
         * This operation takes input's shape, and returns new NDArray filled with ones
         * Expected arguments:
         * input: N-dimensional array
         * 
         */
        DECLARE_OP(ones_as, 1, 1, false);

        /**
         * This operation applies element-wise pow(x, 2) to the given input
         * Expected arguments:
         * input: N-Dimensional array
         */
        DECLARE_OP(square, 1, 1, true);
    }
}