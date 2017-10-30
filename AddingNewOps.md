There's multiple different Ops designs supported in libND4j, and in this guide we'll try to explain how to build your very own operation.

# General overview

### XYZ operations

This kind of operations is actually split into multiple subtypes, based on element-access and result type:
- Transform operations: These operations typically take some NDArray in, and change each element independent of others.
- Reduction operations: These operations take some NDArray and dimensions, and return reduced NDArray (or scalar) back. I.e. sum along dimension(s).
- Scalar operations: These operations are similar to transforms, but they only do arithmetic operations, and second operand is scalar. I.e. each element in given NDArray will add given scalar value.
- Pairwise operations:  These operations are between regular transform opeartions and scalar operations. I.e. element-wise addition of two NDArrays.
- Random operations: Most of these operations related to random numbers distributions: Uniform, Gauss, Bernoulli etc.

Despite differences between these operations, they are all using XZ/XYZ three-operand design, where X and Y are inputs, and Z is output.
Data access in these operations is usually trivial, and loop based. I.e. most trivial loop for scalar transform will look like this:
```c++
for (Nd4jIndex i = start; i < end; i++) {
    result[i] = OpType::op(x[i], scalar, extraParams);
}
```

Operation used in this loop is template-driven, and compiled statically. Here's how, `Add` operation will look like:

```c++

template<typename T>
class Add {
public:
    op_def static T op(T d1, T d2) {
	    return d1 + d2;
	}

    // this signature will be used in Scalar loops
	op_def static T op(T d1, T d2, T *params) {
		return d1 + d2;
	}

    // this signature will be used in reductions
	op_def static T op(T d1) {
		return d1;
	}

	// op for MetaOps
	op_def static T op(T d1, T *params) {
		return d1 + params[0];
	}
};
```

This particular operation is used in different XYZ op groups, but you see the idea: element-wise operation, which is invoked on each element in given NDArray.

### Custom operations

Custom operations is the new concept, added recently and mostly suits SameDiff needs.