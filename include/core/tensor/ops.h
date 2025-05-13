#pragma once
#ifndef TENSOR_OPS_H
#include "core/tensor/tensor.h"

namespace torchscratch::core::tensor {

// Core tensor operations
// Addition: Element-wise addition of two tensors
Tensor add(const Tensor& a, const Tensor& b);

// Multiplication: Element-wise multiplication of two tensors
Tensor mul(const Tensor& a, const Tensor& b);

// Matrix Multiplication: Matrix product of two tensors
Tensor matmul(const Tensor& a, const Tensor& b);

// Transpose: Swap two dimensions of a tensor
Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1);

}  // namespace torchscratch::core::tensor

#endif