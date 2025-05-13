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
  bool owns_data_;                // Ownership of data

  TensorImpl(const std::vector<int64_t>& shape, DType* dtype)
      : data_(nullptr),
        shape_(shape),
        strides_(compute_strides(shape)),
        dtype_(dtype),
        is_contiguous_(true),
        owns_data_(true) {}

  // Copy constructor - Create a shallow copy (share data pointer)
  TensorImpl(const TensorImpl& other)
      : data_(other.data_),  // Share data pointer (shallow copy)
        shape_(other.shape_),
        strides_(other.strides_),
        dtype_(other.dtype_),
        is_contiguous_(other.is_contiguous_),
        owns_data_(false) {  // Mark as non-owning
    // No deep copy of data
  }

  ~TensorImpl() {
    if (owns_data_ && data_) {
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

  // Helper method to get element at specified indices
  template <typename T>
  T& get(const std::vector<int64_t>& indices) {
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      offset += indices[i] * strides_[i];
    }
    return static_cast<T*>(data_)[offset];
  }

  template <typename T>
  const T& get(const std::vector<int64_t>& indices) const {
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      offset += indices[i] * strides_[i];
    }
    return static_cast<const T*>(data_)[offset];
  }
};

// Tensor methods
void Tensor::set_data_ptr(void* data) {
  if (impl_) {
    impl_->data_ = data;
    impl_->owns_data_ = false;  // When setting data externally, mark as non-owning
  }
}

void Tensor::set_strides(const std::vector<int64_t>& strides) {
  if (impl_) {
    impl_->strides_ = strides;
  }
}

void Tensor::set_contiguous(bool is_contiguous) {
  if (impl_) {
    impl_->is_contiguous_ = is_contiguous;
  }
}
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
  Tensor result(new_shape);
  result.set_data_ptr(data_ptr());  // Shallow copy of data
  result.set_strides(
      TensorImpl::compute_strides(new_shape));  // Reshaped tensor may not be contiguous
  result.impl_->owns_data_ = false;             // No ownership transfer
  result.set_contiguous(false);
  return result;
}

Tensor Tensor::clone() const {
  if (!impl_) {
    return Tensor();
  }
  Tensor result(shape());
  result.allocate();
  std::memcpy(result.data_ptr(), data_ptr(), numel() * sizeof(float));
  return result;
}

bool Tensor::is_contiguous() const { return impl_ ? impl_->is_contiguous_ : false; }

// Explicit template instantiations (for common types, to be expanded)
template float* Tensor::data_ptr<float>() const;
template double* Tensor::data_ptr<double>() const;
template int32_t* Tensor::data_ptr<int32_t>() const;

}  // namespace torchscratch::core::tensor
