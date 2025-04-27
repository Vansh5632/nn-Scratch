#pragma once
#ifndef TENSOR_H
#define TENSOR_H

#include <cstdint>
#include <memory>
#include <vector>

namespace torchscratch {
namespace core {
namespace tensor {

class TensorImpl;
class DType;

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

private:
  std::unique_ptr<TensorImpl> impl_;
};

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
