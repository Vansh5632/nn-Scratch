#include "core/tensor/tensor_impl.h"

namespace torchscratch {
namespace core {
namespace tensor {

TensorImpl::TensorImpl(const std::vector<int64_t>& shape, DType* dtype)
    : data_(nullptr),
      shape_(shape),
      strides_(compute_strides(shape)),
      dtype_(dtype),
      is_contiguous_(true),
      owns_data_(true) {}

TensorImpl::TensorImpl(const TensorImpl& other)
    : data_(other.data_),  // Share data pointer (shallow copy)
      shape_(other.shape_),
      strides_(other.strides_),
      dtype_(other.dtype_),
      is_contiguous_(other.is_contiguous_),
      owns_data_(false) {  // Mark as non-owning
  // No deep copy of data
}

TensorImpl::TensorImpl(TensorImpl&& other) noexcept
    : data_(other.data_),
      shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      dtype_(other.dtype_),
      is_contiguous_(other.is_contiguous_),
      owns_data_(other.owns_data_) {
  other.data_ = nullptr;
  other.owns_data_ = false;
}

TensorImpl& TensorImpl::operator=(const TensorImpl& other) {
  if (this != &other) {
    if (owns_data_ && data_) {
      ::operator delete(data_);
    }

    data_ = other.data_;
    shape_ = other.shape_;
    strides_ = other.strides_;
    dtype_ = other.dtype_;
    is_contiguous_ = other.is_contiguous_;
    owns_data_ = false;  // Mark as non-owning for assignment
  }
  return *this;
}

TensorImpl& TensorImpl::operator=(TensorImpl&& other) noexcept {
  if (this != &other) {
    if (owns_data_ && data_) {
      ::operator delete(data_);
    }

    data_ = other.data_;
    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    dtype_ = other.dtype_;
    is_contiguous_ = other.is_contiguous_;
    owns_data_ = other.owns_data_;

    other.data_ = nullptr;
    other.owns_data_ = false;
  }
  return *this;
}

TensorImpl::~TensorImpl() {
  if (owns_data_ && data_) {
    ::operator delete(data_);
  }
}

std::vector<int64_t> TensorImpl::compute_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 0);
  int64_t stride = 1;  // Assuming element size of 1 byte for now (dtype placeholder)
  for (int i = shape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

}  // namespace tensor
}  // namespace core
}  // namespace torchscratch