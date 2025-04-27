#include "core/tensor/tensor.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace torchscratch::core::tensor {

// Internal implementation struct for Tensor (PImpl idiom)
struct TensorImpl {
  void* data_;                    // Raw data pointer
  std::vector<int64_t> shape_;    // Tensor shape
  std::vector<int64_t> strides_;  // Strides (bytes between elements)
  DType* dtype_;                  // Placeholder for data type
  bool is_contiguous_;            // Contiguity flag

  TensorImpl(const std::vector<int64_t>& shape, DType* dtype)
      : data_(nullptr),
        shape_(shape),
        strides_(compute_strides(shape)),
        dtype_(dtype),
        is_contiguous_(true) {}

  // Copy constructor
  TensorImpl(const TensorImpl& other)
      : shape_(other.shape_),
        strides_(other.strides_),
        dtype_(other.dtype_),  // Deep copy if dtype_ is dynamically allocated
        is_contiguous_(other.is_contiguous_) {
    // Deep copy of data if available
    if (other.data_) {
      size_t size =
          std::accumulate(shape_.begin(), shape_.end(), int64_t(1), std::multiplies<int64_t>());
      data_ = ::operator new(size);  // Assuming 1 byte per element
      std::memcpy(data_, other.data_, size);
    } else {
      data_ = nullptr;
    }
  }

  ~TensorImpl() {
    if (data_) {
      ::operator delete(data_);
    }
  }

  static std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 0);
    int64_t stride = 1;  // Assuming element size of 1 byte for now (dtype placeholder)
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }
};

// Tensor constructors
Tensor::Tensor(const std::vector<int64_t>& shape, DType* dtype) {
  impl_ = std::make_unique<TensorImpl>(shape, dtype);
}

Tensor::Tensor(void* data, const std::vector<int64_t>& shape, DType* dtype) {
  impl_ = std::make_unique<TensorImpl>(shape, dtype);
  impl_->data_ = data;
  impl_->is_contiguous_ = true;  // Assume contiguous for externally provided data
}

// Copy constructor
Tensor::Tensor(const Tensor& other) {
  if (other.impl_) {
    impl_ = std::make_unique<TensorImpl>(*other.impl_);
  }
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept : impl_(std::move(other.impl_)) { other.impl_ = nullptr; }

// Copy assignment
Tensor& Tensor::operator=(const Tensor& other) {
  if (this != &other) {
    if (other.impl_) {
      impl_ = std::make_unique<TensorImpl>(*other.impl_);
    } else {
      impl_.reset();
    }
  }
  return *this;
}

// Move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    impl_ = std::move(other.impl_);
    other.impl_ = nullptr;
  }
  return *this;
}

// Destructor
Tensor::~Tensor() = default;

// Shape and stride accessors
const std::vector<int64_t>& Tensor::shape() const {
  static const std::vector<int64_t> empty;
  return impl_ ? impl_->shape_ : empty;
}

const std::vector<int64_t>& Tensor::strides() const {
  static const std::vector<int64_t> empty;
  return impl_ ? impl_->strides_ : empty;
}

int64_t Tensor::dim() const { return impl_ ? static_cast<int64_t>(impl_->shape_.size()) : 0; }

int64_t Tensor::numel() const {
  if (!impl_)
    return 0;
  return std::accumulate(impl_->shape_.begin(), impl_->shape_.end(), int64_t(1),
                         std::multiplies<int64_t>());
}

// Data access
void* Tensor::data_ptr() const { return impl_ ? impl_->data_ : nullptr; }

template <typename T>
T* Tensor::data_ptr() const {
  return impl_ ? static_cast<T*>(impl_->data_) : nullptr;
}

// Memory management
void Tensor::allocate() {
  if (!impl_ || impl_->data_) {
    return;  // Already allocated or invalid
  }
  size_t size = numel() * 1;            // Placeholder: 1 byte per element (dtype TBD)
  impl_->data_ = ::operator new(size);  // Raw allocation (replace with allocator in future)
  impl_->is_contiguous_ = true;
}

void Tensor::deallocate() {
  if (impl_ && impl_->data_) {
    ::operator delete(impl_->data_);
    impl_->data_ = nullptr;
    impl_->is_contiguous_ = false;
  }
}

// Utility methods
Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
  if (!impl_) {
    throw std::runtime_error("Cannot reshape uninitialized tensor");
  }
  int64_t new_numel =
      std::accumulate(new_shape.begin(), new_shape.end(), int64_t(1), std::multiplies<int64_t>());
  if (new_numel != numel()) {
    throw std::runtime_error("Total elements must remain the same for reshape");
  }
  Tensor result(new_shape, impl_->dtype_);
  result.impl_->data_ = impl_->data_;    // Shallow copy of data
  result.impl_->is_contiguous_ = false;  // Reshaped tensor may not be contiguous
  return result;
}

Tensor Tensor::clone() const {
  if (!impl_) {
    return Tensor();
  }
  Tensor result(impl_->shape_, impl_->dtype_);
  result.allocate();
  std::memcpy(result.data_ptr(), impl_->data_, numel() * 1);
  return result;
}

bool Tensor::is_contiguous() const { return impl_ ? impl_->is_contiguous_ : false; }

// Explicit template instantiations (for common types, to be expanded)
template float* Tensor::data_ptr<float>() const;
template double* Tensor::data_ptr<double>() const;
template int32_t* Tensor::data_ptr<int32_t>() const;

}  // namespace torchscratch::core::tensor
