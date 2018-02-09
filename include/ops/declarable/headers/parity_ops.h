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

        /**
        * This op calculates regularized incomplete beta integral Ix(a, b).
        * Implementation is based on two algorithms depending on input values of a and b:
        * - when a and b are both >  maxValue (3000.), then apply Gauss-Legendre quadrature method
        * - when a and b are both <= maxValue (3000.), then apply modified Lentz’s algorithm for continued fractions
        *
        * Input arrays:
        *    a: define power t^{a-1}, must be > 0, type float.
        *    b: define power (1-t)^{b-1}, must be > 0, type float.
        *    x: define upper limit of integration, must be within (0 <= x <= 1) range, type float.
        *
        * Output array:
        *    0: values of  regularized incomplete beta integral that corresponds to variable upper limit x, type float
        *
        * Three input and one output arrays must have the same shape
        */
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
        DECLARE_CONFIGURABLE_OP(fill_as, 1, 1, true, 0, 0);

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
         * This operation calculate the confusion matrix for a
         * pair of prediction and label 1-D arrays.
         * Expected arguments:
         * Input arrays:
         *   0 - predictions: 1-D array
         *   1 - labels: 1-D array
         *   2 - weights : optional
         * Int args:
         *   0 - num_classes: optional
         *
         */
        DECLARE_CUSTOM_OP(confusion_matrix, 2, 1, false, 0, -2);

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

        /**
        * This op calculates Hurwitz zeta function zeta(x, q) = sum_{n=0}^{inf} (q + n)^{-x}
        * Implementation is based on Euler-Maclaurin summation formula
        *
        *   Input arrays:
        *   x: define power {-x}, must be > 1, type float.
        *   q: define summand in denominator, must be > 0, type float.
        *
        * Output array:
        *    0: corresponding values of Hurwitz zeta function
        *
        * Two input and one output arrays must have the same shape
        */
        DECLARE_CONFIGURABLE_OP(zeta, 2, 1, false, 0, 0);

        /**
        * This op calculates polygamma function psi^(n)(x). Implementation is based on serial representation written in
        * terms of the Hurwitz zeta function: polygamma = (-1)^{n+1} * n! * zeta(n+1, x).
        * Currently the case n = 0 is not supported.
        *
        * Input arrays:
        *    0: n - define derivative order (n+1), type integer (however currently is implemented as float casted to integer)
        *    1: x - abscissa points where to evaluate the polygamma function, type float
        *
        * Output array:
        *    0: values of polygamma function at corresponding x, type float
        *
        * Two input and one output arrays have the same shape
        */
        DECLARE_CONFIGURABLE_OP(polygamma, 2, 1, false, 0, 0);

        /**
         * This operation takes shape as first argument, and returns new NDArray filled with specific scalar value.
         * Input arrays:
         * 0 - shape vector
         * 1 - optional scalar NDArray
         * 
         * T arguments:
         * 0 - optional scalar value
         * 
         */
        DECLARE_CUSTOM_OP(fill, 1, 1, false, -2, 0);

        /**
         * This operation splits given NDArray into chunks of specific size, along given dimension
         * Input arrays:
         * 0 - input array
         * 1 - array of sizes
         * 2 - optional axis
         * 
         * Integer arguments:
         * 0 - optional axis
         * 
         */
        DECLARE_CUSTOM_OP(split_v, 2, -1, false, 0, -2);

        /**
         * This operation splits given NDArray into chunks of specific size, along given dimension
         * 0 - input array
         * 1 - optional axis
         * 
         * Integer arguments:
         * 0 - number of splits
         * 1 - optional axis
         */
        DECLARE_CUSTOM_OP(split, 1, -1, false, 0, 1);


        /**
         * This operation adjusts image hue by delta
         * Input arrays:
         * 0 - 1D or 3D input array, must have 3 channels.
         * 1 - optional scalar, delta value
         * 
         * T arguments:
         * 0 - optional delta value
         * 
         * Int arguments:
         * 0 - optional argument, isNHWC. false by default.
         */
        DECLARE_CONFIGURABLE_OP(adjust_hue, 1, 1, true, -2, -2);

        /**
         * This operation adjusts image saturation by delta
         * Input arrays:
         * 0 - 1D or 3D input array, must have 3 channels.
         * 1 - optional scalar, delta value
         * 
         * T arguments:
         * 0 - optional delta value
         * 
         * Int arguments:
         * 0 - optional argument, isNHWC. false by default.
         */
        DECLARE_CONFIGURABLE_OP(adjust_saturation, 1, 1, true, -2, -2);


        /**
         * 
         * 
         *
         */
        DECLARE_CUSTOM_OP(depth_to_space, 1, 1, false, 0, 2);

        /**
         * 
         * 
         *
         */
        DECLARE_CUSTOM_OP(space_to_depth, 1, 1, false, 0, 2);

        /**
         * This op calculates cross-product between input arguments
         * Input arguments
         * 0 - vector or tensor A
         * 1 - vector or tensor B
         */
        DECLARE_OP(cross, 2, 1, false);

        /**
         * 
         * 
         */
        DECLARE_CUSTOM_OP(space_to_batch, 1, 1, false, 0, -2);

        /**
         * 
         * 
         */
        DECLARE_CUSTOM_OP(batch_to_space, 1, 1, false, 0, -2);

        /**
         * top_k operation returns a vector of k top values for 
         *  given NDArray as tensor with default boolean (true)
         *  as sort for result index array
         *  will be sorted by the values in descending order.
         *  The first parameter is a NDArray for working.
         *  The second is k (default 1) - optional
         *  The third is boolean value(default is 1) (0 - as is, 1 - sorted by value) optional
         */
        DECLARE_CUSTOM_OP(top_k, 1, 2, false, 0, -2);

        /**
         * in_top_k operation returns a vector of k boolean values for 
         *  given NDArray as 2D matrix of predicted in the NDArray k top values
         *  The first parameter is a NDArray of predicted values (2d array).
         *  The second is NDArray as vector of indeces k top values will be search.
         *  The third is k
         */

        DECLARE_CUSTOM_OP(in_top_k, 2, 1, true, 1, 1);

        /**
         * moments operation calculate a mean and variation for given NDArray
         * with reduce a result according to axis array given.
         * For full axis the result is both mean and variance of all members in array.
         * Otherwise there are two NDArrays with means and variances for 
         * Axes can be put as the second NDArray or as int vector.
         */
        DECLARE_CUSTOM_OP(moments, 1, 2, false, 0, -2);

        /**
         * embedding_lookup - search for submatrices in given matrix and retunts them
         * accordingly to index array given.
         */
        DECLARE_CUSTOM_OP(embedding_lookup, 2, 1, false, 0, 1);

        /**
         * dynamic_partition - partition a input tensor onto num_partitions 
         * accordingly to index array given.
         *
         * the first param - NDArray to be partitioned.
         * the second param - index array
         * the third param (integer param) - num or partitions.
         * 
         * returns a num of NDArrays as output
         */

        DECLARE_CUSTOM_OP(dynamic_partition, 2, 1, false, 0, 1);

        /**
         * dynamic_stitch - merge partitions from the second param a input tensor 
         * into a single tensor accordingly to index array given.
         *
         * the first param - index array
         * the second params - tensors to be merged
         * 
         * returns a num of NDArrays as output
         * 
         * the operation is inversion od dynamic_partition
         */
        DECLARE_CUSTOM_OP(dynamic_stitch, 2, 1, false, 0, 0);

        /**
         * zero_fraction op.
         * compute a fraction of zeros in given array
         *
         * input param - an array (tensor)
         * output value - a real number with given type (e.g. float or double)
         */
        DECLARE_CUSTOM_OP(zero_fraction, 1, 1, false, 0, 0);

        /**
         * xw_plus_b op.
         * multiply two first matrices and add third vector to each row of result
         *
         * input params:
         *   - 2D matrix NxM
         *   - 2D matrix MxN
         *   - 1D vector with N elements
         * output value - 2D matrix NxN as multiply of matrixes and add vector
         */
        DECLARE_CUSTOM_OP(xw_plus_b, 3, 1, false, 0, 0);

        /**
         * This operation is missed due it simplicy.
         * Input and output params are the same after operation.
         * Input - NDArray, output - NDArray with the same shape.
         */
        DECLARE_OP(stop_gradient, 1, 1, true);

        /**
         * l2_loss op.
         * compute a l2 norm for given array.
         *
         * input param - an array (tensor)
         * output value - a real number with given type (e.g. float or double)
         */
        DECLARE_CUSTOM_OP(l2_loss, 1, 1, false, 0, 0);

        DECLARE_CUSTOM_OP(parallel_stack, -1, 1, false, 0, 0);

	/**
         * This op calculates logarithmic loss of poison distributed input
         * Input arguments
         *  0 - target
         *  1 - input
         *  optional int - boolean value compute_full_loss: 0 (default) or 1 (compute)
         */
        DECLARE_CONFIGURABLE_OP(log_poison_loss, 2, 1, true, 0, 0);

        /**
         * normalize_moments operation normalize already calculated mean and variation 
         * accordingly to shift and count.
         * input params:
         *  - count of data
         *  - tensor with mean
         *  - tensor with variance (the same shape as before)
         *
         *  - optional floating point param shift.
         * 
         *  returns a normalized pair mean and variance with the same shapes as input
         */
        DECLARE_CUSTOM_OP(normalize_moments, 3, 2, false, 1, 0);


    }
}