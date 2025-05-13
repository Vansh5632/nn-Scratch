#include "core/tensor/ops.h"

#include <algorithm>
#include <stdexcept>

namespace torchscratch::core::tensor {

Tensor add(const Tensor& a, const Tensor& b) {
  // Basic element-wise addition
  if (!a.data_ptr() || !b.data_ptr()) {
    throw std::runtime_error("Input tensors must have allocated data");
  }

  // Handle broadcasting for scalar case
  if (b.numel() == 1) {
    Tensor result(a.shape());
    result.allocate();
    float scalar = *b.data_ptr<float>();
    float* a_data = a.data_ptr<float>();
    float* result_data = result.data_ptr<float>();

    for (int i = 0; i < a.numel(); ++i) {
      result_data[i] = a_data[i] + scalar;
    }
    return result;
  }

  // Regular element-wise addition
  if (a.shape() != b.shape()) {
    throw std::runtime_error("Tensor shapes must match for addition");
  }

  Tensor result(a.shape());
  result.allocate();

  float* a_data = a.data_ptr<float>();
  float* b_data = b.data_ptr<float>();
  float* result_data = result.data_ptr<float>();

  for (int i = 0; i < a.numel(); ++i) {
    result_data[i] = a_data[i] + b_data[i];
  }

  return result;
}

Tensor mul(const Tensor& a, const Tensor& b) {
  if (!a.data_ptr() || !b.data_ptr()) {
    throw std::runtime_error("Input tensors must have allocated data");
  }

  if (a.shape() != b.shape()) {
    throw std::runtime_error("Tensor shapes must match for element-wise multiplication");
  }

  Tensor result(a.shape());
  result.allocate();

  float* a_data = a.data_ptr<float>();
  float* b_data = b.data_ptr<float>();
  float* result_data = result.data_ptr<float>();

  for (int i = 0; i < a.numel(); ++i) {
    result_data[i] = a_data[i] * b_data[i];
  }

  return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  if (!a.data_ptr() || !b.data_ptr()) {
    throw std::runtime_error("Input tensors must have allocated data");
  }

  if (a.dim() != 2 || b.dim() != 2) {
    throw std::runtime_error("Both tensors must be 2D for matrix multiplication");
  }

  auto a_shape = a.shape();
  auto b_shape = b.shape();

  if (a_shape[1] != b_shape[0]) {
    throw std::runtime_error("Inner dimensions must match for matrix multiplication");
  }

  int m = a_shape[0];
  int k = a_shape[1];
  int n = b_shape[1];

  Tensor result({m, n});
  result.allocate();

  float* a_data = a.data_ptr<float>();
  float* b_data = b.data_ptr<float>();
  float* result_data = result.data_ptr<float>();

  // Simple matrix multiplication (this can be optimized)
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int p = 0; p < k; ++p) {
        sum += a_data[i * k + p] * b_data[p * n + j];
      }
      result_data[i * n + j] = sum;
    }
  }

  return result;
}

Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1) {
  if (!a.data_ptr()) {
    throw std::runtime_error("Input tensor must have allocated data");
  }

  if (dim0 < 0 || dim0 >= a.dim() || dim1 < 0 || dim1 >= a.dim()) {
    throw std::runtime_error("Invalid dimensions for transpose");
  }

  auto shape = a.shape();
  auto strides = a.strides();
  std::swap(shape[dim0], shape[dim1]);

  // Important: Swap strides to reflect the new memory layout
  std::swap(strides[dim0], strides[dim1]);

  // Create the transposed tensor
  Tensor result(shape);
  result.set_data_ptr(a.data_ptr());  // Shallow copy (view)
  result.set_strides(strides);
  result.set_contiguous(false);  // Transpose creates non-contiguous tensor

  return result;
}

}  // namespace torchscratch::core::tensor