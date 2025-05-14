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
class Variable;

class Function {
public:
  virtual ~Function() = default;
  Function() = default;
  Function(const Function&) = default;
  Function& operator=(const Function&) = default;
  Function(Function&&) noexcept = default;
  Function& operator=(Function&&) noexcept = default;

  virtual std::vector<tensor::Tensor> forward(const std::vector<tensor::Tensor>& inputs) = 0;

  virtual std::vector<tensor::Tensor> backward(const std::vector<tensor::Tensor>& grad_outputs) = 0;

  virtual std::string name() const = 0;

  void save_for_backward(const std::vector<tensor::Tensor>& inputs);

  const std::vector<Variable*>& get_saved_variables() const;

private:
  std::vector<Variable*> saved_variables_;
};
class AddFunction:public Function{
    public:
        std::vector<tensor::Tensor> forward(const std::vector<tensor::Tensor>& inputs) override;
        std::vector<tensor::Tensor> backward(const std::vector<tensor::Tensor>& grad_outputs) override;
        std::string name() const override {
            return "AddFunction";
        }
};
class MulFunction:public Function{
    public:
        std::vector<tensor::Tensor> forward(const std::vector<tensor::Tensor>& inputs) override;
        std::vector<tensor::Tensor> backward(const std::vector<tensor::Tensor>& grad_outputs) override;
        std::string name() const override {
            return "MulFunction";
        }
    private:
        tensor::Tensor input1_;
        tensor::Tensor input2_;
};
class MatMulFunction:public Function{
    public:
        std::vector<tensor::Tensor> forward(const std::vector<tensor::Tensor>& inputs) override;
        std::vector<tensor::Tensor> backward(const std::vector<tensor::Tensor>& grad_outputs) override;
        std::string name() const override {
            return "MatMulFunction";
        }
    private:
        tensor::Tensor input1_;
        tensor::Tensor input2_;
};

}  // namespace autograd
}  // namespace core
}  // namespace torchscratch
#endif