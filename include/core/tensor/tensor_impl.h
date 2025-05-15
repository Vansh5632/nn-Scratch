#pragma once
#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include <cstddef>  // For std::size_t
#include <cstdint>
#include <vector>

namespace torchscratch {
namespace core {
namespace tensor {

class DType;

/**
 * Internal implementation for Tensor class.
 */
struct TensorImpl {
  void* data_ = nullptr;          // Raw data pointer
  std::vector<int64_t> shape_;    // Tensor shape
  std::vector<int64_t> strides_;  // Strides (bytes between elements)
  DType* dtype_ = nullptr;        // Placeholder for data type
  bool is_contiguous_ = true;     // Contiguity flag
  bool owns_data_ = true;         // Ownership of data

  TensorImpl() = default;

  TensorImpl(const std::vector<int64_t>& shape, DType* dtype);

  // Copy constructor - Create a shallow copy (share data pointer)
  TensorImpl(const TensorImpl& other);

  // Move constructor
  TensorImpl(TensorImpl&& other) noexcept;

  // Assignment operators
  TensorImpl& operator=(const TensorImpl& other);
  TensorImpl& operator=(TensorImpl&& other) noexcept;

  ~TensorImpl();

  static std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape);

  // Helper method to get element at specified indices
  template <typename T>
  T& get(const std::vector<int64_t>& indices) {
    std::size_t offset = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
      offset += indices[i] * strides_[i];
    }
    return static_cast<T*>(data_)[offset];
  }

  template <typename T>
  const T& get(const std::vector<int64_t>& indices) const {
    std::size_t offset = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
      offset += indices[i] * strides_[i];
    }
    return static_cast<const T*>(data_)[offset];
  }
};

}  // namespace tensor
}  // namespace core
}  // namespace torchscratch

#endif  // TENSOR_IMPL_H