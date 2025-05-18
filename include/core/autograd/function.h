#pragma once
#ifndef AUTOGRAD_FUNCTION_H
#define AUTOGRAD_FUNCTION_H

#include <memory>
#include <string>
#include <vector>

#include "core/tensor/tensor.h"

namespace torchscratch {
namespace core {
namespace autograd {

// Forward declaration
class Variable;

/**
 * Base class for all autograd functions.
 * Each subclass implements a particular operation (like Add, MatMul, etc.)
 * with its forward and backward pass logic.
 */
class Function {
public:
  virtual ~Function() = default;

  // Rule of 5: Allow proper inheritance
  Function() = default;
  Function(const Function&) = default;
  Function& operator=(const Function&) = default;
  Function(Function&&) noexcept = default;
  Function& operator=(Function&&) noexcept = default;

  /**
   * Apply the forward pass of this function to the inputs.
   * @param inputs Vector of input tensors
   * @return Vector of output tensors
   */
  virtual std::vector<tensor::Tensor> forward(const std::vector<tensor::Tensor>& inputs) = 0;

  /**
   * Apply the backward pass of this function.
   * @param grad_output Gradient of the loss with respect to the output
   * @return Vector of gradients with respect to the inputs
   */
  virtual std::vector<tensor::Tensor> backward(const std::vector<tensor::Tensor>& grad_output) = 0;

  /**
   * Get a string representation of this function for debugging.
   */
  virtual std::string name() const = 0;

  /**
   * Save pointers to the input variables for backward pass.
   * @param inputs Vector of input variables
   */
  void save_for_backward(const std::vector<Variable*>& inputs);

  /**
   * Get the saved input variables.
   * @return Vector of saved input variables
   */
  const std::vector<Variable*>& get_saved_variables() const;

private:
  std::vector<Variable*> saved_variables_;
};

/**
 * AddFunction implements element-wise addition with broadcasting.
 */
class AddFunction : public Function {
public:
  std::vector<tensor::Tensor> forward(const std::vector<tensor::Tensor>& inputs) override;
  std::vector<tensor::Tensor> backward(const std::vector<tensor::Tensor>& grad_output) override;
  std::string name() const override { return "AddFunction"; }
};

/**
 * MulFunction implements element-wise multiplication.
 */
class MulFunction : public Function {
public:
  std::vector<tensor::Tensor> forward(const std::vector<tensor::Tensor>& inputs) override;
  std::vector<tensor::Tensor> backward(const std::vector<tensor::Tensor>& grad_output) override;
  std::string name() const override { return "MulFunction"; }

private:
  tensor::Tensor input1_; // Save inputs for backward pass
  tensor::Tensor input2_;
};

/**
 * MatMulFunction implements matrix multiplication.
 */
class MatMulFunction : public Function {
public:
  std::vector<tensor::Tensor> forward(const std::vector<tensor::Tensor>& inputs) override;
  std::vector<tensor::Tensor> backward(const std::vector<tensor::Tensor>& grad_output) override;
  std::string name() const override { return "MatMulFunction"; }

private:
  tensor::Tensor input1_; // Save inputs for backward pass
  tensor::Tensor input2_;
};

} // namespace autograd
} // namespace core
} // namespace torchscratch

#endif // AUTOGRAD_FUNCTION_H