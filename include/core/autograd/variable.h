#pragma once
#ifndef AUTOGRAD_VARIABLE_H
#define AUTOGRAD_VARIABLE_H

#include <memory>
#include <vector>

#include "core/autograd/function.h"
#include "core/tensor/tensor.h"

namespace torchscratch {
namespace core {
namespace autograd {

/**
 * Variable wraps a Tensor and tracks gradient information for automatic differentiation.
 * It represents a node in the computational graph.
 */
class Variable {
public:
  /**
   * Create a variable from a tensor.
   * @param data The tensor data
   * @param requires_grad Whether to track gradients for this variable
   */
  explicit Variable(const tensor::Tensor& data, bool requires_grad = false);

  /**
   * Create a detached copy of this variable.
   * @return A new variable with the same data but no gradient tracking
   */
  Variable detach() const;

  /**
   * Get the underlying tensor data.
   */
  const tensor::Tensor& data() const { return data_; }

  /**
   * Get the gradient tensor.
   */
  const tensor::Tensor& grad() const { return grad_; }

  /**
   * Set the gradient tensor.
   */
  void set_grad(const tensor::Tensor& grad) { grad_ = grad; }

  /**
   * Check if this variable requires gradient computation.
   */
  bool requires_grad() const { return requires_grad_; }

  /**
   * Set whether this variable requires gradient computation.
   */
  void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }

  /**
   * Get the gradient function that created this variable.
   */
  std::shared_ptr<Function> grad_fn() const { return grad_fn_; }

  /**
   * Set the gradient function for this variable.
   */
  void set_grad_fn(std::shared_ptr<Function> grad_fn) { grad_fn_ = grad_fn; }

  /**
   * Start backpropagation from this variable.
   */
  void backward();

  /**
   * Get the shape of the underlying tensor.
   */
  const std::vector<int64_t>& shape() const { return data_.shape(); }

  /**
   * Get the number of dimensions of the underlying tensor.
   */
  int64_t dim() const { return data_.dim(); }

  /**
   * Get the total number of elements in the underlying tensor.
   */
  int64_t numel() const { return data_.numel(); }

private:
  tensor::Tensor data_;                // The tensor data
  tensor::Tensor grad_;                // Gradient with respect to this variable
  bool requires_grad_;                 // Whether to track gradients for this variable
  std::shared_ptr<Function> grad_fn_;  // The function that created this variable
};

/**
 * Element-wise addition of two variables.
 */
Variable add(const Variable& a, const Variable& b);

/**
 * Element-wise multiplication of two variables.
 */
Variable mul(const Variable& a, const Variable& b);

/**
 * Matrix multiplication of two variables.
 */
Variable matmul(const Variable& a, const Variable& b);

}  // namespace autograd
}  // namespace core
}  // namespace torchscratch

#endif  // AUTOGRAD_VARIABLE_H