#include "core/nn/loss.h"

#include <cmath>

#include "core/autograd/function.h"
#include "core/tensor/ops.h"

namespace torchscratch {
namespace core {
namespace nn {

// MSE Loss Function
class MSELossFunction : public autograd::Function {
public:
  std::vector<tensor::Tensor> forward(const std::vector<tensor::Tensor>& inputs) override {
    const tensor::Tensor& predicted = inputs[0];
    const tensor::Tensor& target = inputs[1];

    // Compute (predicted - target)^2
    tensor::Tensor diff = tensor::sub(predicted, target);
    tensor::Tensor squared_diff = tensor::mul(diff, diff);

    // Mean over all elements
    float sum = 0.0f;
    const float* data = squared_diff.data_ptr<float>();
    for (int64_t i = 0; i < squared_diff.numel(); ++i) {
      sum += data[i];
    }

    tensor::Tensor loss({1});
    loss.allocate();
    float* loss_data = loss.data_ptr<float>();
    loss_data[0] = sum / static_cast<float>(squared_diff.numel());

    return {loss};
  }

  std::vector<tensor::Tensor> backward(const std::vector<tensor::Tensor>& grad_output) override {
    const auto& saved_vars = get_saved_variables();
    const tensor::Tensor& predicted = saved_vars[0]->data();
    const tensor::Tensor& target = saved_vars[1]->data();
    const tensor::Tensor& grad_out = grad_output[0];

    // Gradient w.r.t predicted: 2 * (predicted - target) / N
    tensor::Tensor grad_predicted = tensor::sub(predicted, target);

    float scale = 2.0f / static_cast<float>(predicted.numel());
    float* grad_data = grad_predicted.data_ptr<float>();
    for (int64_t i = 0; i < grad_predicted.numel(); ++i) {
      grad_data[i] *= scale;
    }

    // Gradient w.r.t target: -2 * (predicted - target) / N
    tensor::Tensor grad_target(target.shape());
    grad_target.allocate();
    float* grad_target_data = grad_target.data_ptr<float>();
    for (int64_t i = 0; i < grad_target.numel(); ++i) {
      grad_target_data[i] = -grad_data[i];
    }

    return {grad_predicted, grad_target};
  }

  std::string name() const override { return "MSELossFunction"; }
};

// Binary Cross Entropy Loss Function
class BCELossFunction : public autograd::Function {
public:
  std::vector<tensor::Tensor> forward(const std::vector<tensor::Tensor>& inputs) override {
    const tensor::Tensor& predicted = inputs[0];
    const tensor::Tensor& target = inputs[1];

    // BCE = -[target * log(predicted) + (1 - target) * log(1 - predicted)]
    float sum = 0.0f;
    const float* pred_data = predicted.data_ptr<float>();
    const float* target_data = target.data_ptr<float>();

    const float eps = 1e-8f;  // For numerical stability

    for (int64_t i = 0; i < predicted.numel(); ++i) {
      float p = std::max(eps, std::min(1.0f - eps, pred_data[i]));
      float t = target_data[i];
      sum += -(t * std::log(p) + (1.0f - t) * std::log(1.0f - p));
    }

    tensor::Tensor loss({1});
    loss.allocate();
    float* loss_data = loss.data_ptr<float>();
    loss_data[0] = sum / static_cast<float>(predicted.numel());

    return {loss};
  }

  std::vector<tensor::Tensor> backward(const std::vector<tensor::Tensor>& grad_output) override {
    const auto& saved_vars = get_saved_variables();
    const tensor::Tensor& predicted = saved_vars[0]->data();
    const tensor::Tensor& target = saved_vars[1]->data();
    const tensor::Tensor& grad_out = grad_output[0];

    // Gradient w.r.t predicted: -(target/predicted - (1-target)/(1-predicted)) / N
    tensor::Tensor grad_predicted(predicted.shape());
    grad_predicted.allocate();

    const float* pred_data = predicted.data_ptr<float>();
    const float* target_data = target.data_ptr<float>();
    float* grad_data = grad_predicted.data_ptr<float>();

    const float eps = 1e-8f;
    float scale = 1.0f / static_cast<float>(predicted.numel());

    for (int64_t i = 0; i < predicted.numel(); ++i) {
      float p = std::max(eps, std::min(1.0f - eps, pred_data[i]));
      float t = target_data[i];
      grad_data[i] = scale * (-(t / p) + (1.0f - t) / (1.0f - p));
    }

    // Gradient w.r.t target (usually not needed for training)
    tensor::Tensor grad_target(target.shape());
    grad_target.allocate();
    // Zero gradient for target

    return {grad_predicted, grad_target};
  }

  std::string name() const override { return "BCELossFunction"; }
};

autograd::Variable mse_loss(const autograd::Variable& predicted, const autograd::Variable& target) {
  auto func = std::make_shared<MSELossFunction>();

  // Forward pass
  std::vector<tensor::Tensor> result_tensors = func->forward({predicted.data(), target.data()});
  tensor::Tensor result_tensor = result_tensors[0];

  bool requires_grad = predicted.requires_grad() || target.requires_grad();
  autograd::Variable result(result_tensor, requires_grad);

  if (requires_grad) {
    result.set_grad_fn(func);
    auto pred_var = const_cast<autograd::Variable*>(&predicted);
    auto target_var = const_cast<autograd::Variable*>(&target);
    func->save_for_backward({pred_var, target_var});
  }

  return result;
}

autograd::Variable binary_cross_entropy_loss(const autograd::Variable& predicted,
                                             const autograd::Variable& target) {
  auto func = std::make_shared<BCELossFunction>();

  // Forward pass
  std::vector<tensor::Tensor> result_tensors = func->forward({predicted.data(), target.data()});
  tensor::Tensor result_tensor = result_tensors[0];

  bool requires_grad = predicted.requires_grad() || target.requires_grad();
  autograd::Variable result(result_tensor, requires_grad);

  if (requires_grad) {
    result.set_grad_fn(func);
    auto pred_var = const_cast<autograd::Variable*>(&predicted);
    auto target_var = const_cast<autograd::Variable*>(&target);
    func->save_for_backward({pred_var, target_var});
  }

  return result;
}

autograd::Variable cross_entropy_loss(const autograd::Variable& predicted,
                                      const autograd::Variable& target) {
  // For now, implement as a placeholder
  // Full cross-entropy would require softmax + log-likelihood
  return binary_cross_entropy_loss(predicted, target);
}

}  // namespace nn
}  // namespace core
}  // namespace torchscratch
