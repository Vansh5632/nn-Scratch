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

  // Handle broadcasting for bias addition (2D + 1D case)
  // a: [batch_size, features], b: [features]
  if (a.dim() == 2 && b.dim() == 1 && a.shape()[1] == b.shape()[0]) {
    Tensor result(a.shape());
    result.allocate();

    float* a_data = a.data_ptr<float>();
    float* b_data = b.data_ptr<float>();
    float* result_data = result.data_ptr<float>();

    int64_t batch_size = a.shape()[0];
    int64_t features = a.shape()[1];

    for (int64_t i = 0; i < batch_size; ++i) {
      for (int64_t j = 0; j < features; ++j) {
        result_data[i * features + j] = a_data[i * features + j] + b_data[j];
      }
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

// Transpose implementation - creates a non-contiguous view
Tensor transpose(const Tensor& a, int dim0, int dim1) {
  if (a.dim() < 2) {
    throw std::runtime_error("Cannot transpose tensor with less than 2 dimensions");
  }

  if (dim0 < 0 || dim0 >= a.dim() || dim1 < 0 || dim1 >= a.dim()) {
    throw std::runtime_error("Transpose dimensions out of range");
  }

  // Create a view with swapped shape
  std::vector<int64_t> out_shape = a.shape();
  std::swap(out_shape[dim0], out_shape[dim1]);

  // Create a new tensor that shares the same data
  Tensor result(a.data_ptr(), out_shape);

  // Compute the transposed strides
  std::vector<int64_t> transposed_strides = a.strides();
  std::swap(transposed_strides[dim0], transposed_strides[dim1]);

  // Set the custom strides and mark as non-contiguous
  result.set_strides(transposed_strides);
  result.set_contiguous(false);

  return result;
}

Tensor sub(const Tensor& a, const Tensor& b) {
  // Basic element-wise subtraction
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
      result_data[i] = a_data[i] - scalar;
    }
    return result;
  }

  // Regular element-wise subtraction
  if (a.shape() != b.shape()) {
    throw std::runtime_error("Tensor shapes must match for subtraction");
  }

  Tensor result(a.shape());
  result.allocate();

  float* a_data = a.data_ptr<float>();
  float* b_data = b.data_ptr<float>();
  float* result_data = result.data_ptr<float>();

  for (int i = 0; i < a.numel(); ++i) {
    result_data[i] = a_data[i] - b_data[i];
  }

  return result;
}
}  // namespace torchscratch::core::tensor