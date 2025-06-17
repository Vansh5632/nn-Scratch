#pragma once
#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "core/tensor/tensor.h"

namespace torchscratch::core::tensor {

// Core tensor operations
// Addition: Element-wise addition of two tensors
Tensor add(const Tensor& a, const Tensor& b);

// Subtraction: Element-wise subtraction of two tensors
Tensor sub(const Tensor& a, const Tensor& b);

// Multiplication: Element-wise multiplication of two tensors
Tensor mul(const Tensor& a, const Tensor& b);

// Matrix multiplication
Tensor matmul(const Tensor& a, const Tensor& b);

// Transpose: Swap two dimensions of a tensor
Tensor transpose(const Tensor& a, int dim0 = 0, int dim1 = 1);

}  // namespace torchscratch::core::tensor

#endif  // TENSOR_OPS_H