#pragma once
#ifndef TENSOR_H
#define TENSOR_H

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "core/tensor/tensor_impl.h"

namespace torchscratch {
namespace core {
namespace tensor {

class TensorImpl;
class DType;

// Forward declaration for the accessor class
template <typename T>
class TransposedTensorAccessor;

class Tensor {
public:
  Tensor() = default;

  explicit Tensor(const std::vector<int64_t>& shape, DType* dtype = nullptr);

  Tensor(void* data, const std::vector<int64_t>& shape, DType* dtype = nullptr);

  Tensor(const Tensor& other);
  Tensor(Tensor&& other) noexcept;

  Tensor& operator=(const Tensor& other);
  Tensor& operator=(Tensor&& other) noexcept;

  ~Tensor();

  const std::vector<int64_t>& shape() const;
  const std::vector<int64_t>& strides() const;

  int64_t dim() const;
  int64_t numel() const;

  void* data_ptr() const;
  template <typename T>
  T* data_ptr() const;

  static bool is_cuda() { return false; }
  void allocate();
  void deallocate();

  Tensor reshape(const std::vector<int64_t>& new_shape) const;
  Tensor clone() const;

  bool is_contiguous() const;

  // New methods to expose TensorImpl functionality
  void set_data_ptr(void* data);
  void set_strides(const std::vector<int64_t>& strides);
  void set_contiguous(bool is_contiguous);

  // Add accessor method for transposed tensors
  template <typename T>
  TransposedTensorAccessor<T> transposed_accessor() const;

  // Friend access for the accessor class
  template <typename T>
  friend class TransposedTensorAccessor;

private:
  std::unique_ptr<TensorImpl> impl_;
};

// Class to provide element-wise access to transposed tensors with proper striding
template <typename T>
class TransposedTensorAccessor {
public:
  TransposedTensorAccessor(const Tensor& tensor) : tensor_(tensor) {}

  // Access elements using row-major indexing but with proper stride handling
  T operator[](size_t idx) const {
    if (!tensor_.data_ptr()) {
      throw std::runtime_error("Cannot access data of uninitialized tensor");
    }

    // For a 2D transposed tensor (most common case)
    if (tensor_.dim() == 2) {
      auto shape = tensor_.shape();
      int64_t rows = shape[0];
      int64_t cols = shape[1];

      int64_t row = idx / cols;
      int64_t col = idx % cols;

      T* data = static_cast<T*>(tensor_.data_ptr());
      auto strides = tensor_.strides();

      // Use strides to calculate the actual memory offset
      size_t offset = row * strides[0] + col * strides[1];
      return data[offset];
    } else {
      throw std::runtime_error("Accessor currently only supports 2D tensors");
    }
  }

private:
  const Tensor& tensor_;
};

// Implementation of the accessor method
template <typename T>
TransposedTensorAccessor<T> Tensor::transposed_accessor() const {
  return TransposedTensorAccessor<T>(*this);
}

class DType {
public:
  virtual ~DType() = default;

  // Rule of 5: Implement copy/move operations
  DType() = default;
  DType(const DType&) = default;
  DType& operator=(const DType&) = default;
  DType(DType&&) noexcept = default;
  DType& operator=(DType&&) noexcept = default;
};

}  // namespace tensor
}  // namespace core
}  // namespace torchscratch

#endif  // TENSOR_H