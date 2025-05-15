#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "core/autograd/variable.h"
#include "core/tensor/ops.h"
#include "core/tensor/tensor_impl.h"

namespace torchscratch {
namespace core {
namespace autograd {

// Function implementation
void Function::save_for_backward(const std::vector<Variable*>& inputs) {
  saved_variables_ = inputs;
}

const std::vector<Variable*>& Function::get_saved_variables() const { return saved_variables_; }

// AddFunction implementation
std::vector<tensor::Tensor> AddFunction::forward(const std::vector<tensor::Tensor>& inputs) {
  if (inputs.size() != 2) {
    throw std::runtime_error("AddFunction expects exactly 2 inputs");
  }
  std::vector<tensor::Tensor> outputs;
  outputs.push_back(tensor::add(inputs[0], inputs[1]));
  return outputs;
}

std::vector<tensor::Tensor> AddFunction::backward(const std::vector<tensor::Tensor>& grad_output) {
  if (grad_output.size() != 1) {
    throw std::runtime_error("AddFunction backward expects exactly 1 gradient");
  }

  // Gradient of addition is passed unchanged to both inputs
  std::vector<tensor::Tensor> grad_inputs;
  grad_inputs.push_back(grad_output[0]);
  grad_inputs.push_back(grad_output[0]);
  return grad_inputs;
}

// MulFunction implementation
std::vector<tensor::Tensor> MulFunction::forward(const std::vector<tensor::Tensor>& inputs) {
  if (inputs.size() != 2) {
    throw std::runtime_error("MulFunction expects exactly 2 inputs");
  }

  // Store inputs for backward pass
  input1_ = inputs[0];
  input2_ = inputs[1];

  std::vector<tensor::Tensor> outputs;
  outputs.push_back(tensor::mul(inputs[0], inputs[1]));
  return outputs;
}

std::vector<tensor::Tensor> MulFunction::backward(const std::vector<tensor::Tensor>& grad_output) {
  if (grad_output.size() != 1) {
    throw std::runtime_error("MulFunction backward expects exactly 1 gradient");
  }

  // d(a*b)/da = b * grad_output
  // d(a*b)/db = a * grad_output
  std::vector<tensor::Tensor> grad_inputs;
  grad_inputs.push_back(tensor::mul(input2_, grad_output[0]));
  grad_inputs.push_back(tensor::mul(input1_, grad_output[0]));
  return grad_inputs;
}

// MatMulFunction implementation
std::vector<tensor::Tensor> MatMulFunction::forward(const std::vector<tensor::Tensor>& inputs) {
  if (inputs.size() != 2) {
    throw std::runtime_error("MatMulFunction expects exactly 2 inputs");
  }

  // Store inputs for backward pass
  input1_ = inputs[0];
  input2_ = inputs[1];

  std::vector<tensor::Tensor> outputs;
  outputs.push_back(tensor::matmul(inputs[0], inputs[1]));
  return outputs;
}

std::vector<tensor::Tensor> MatMulFunction::backward(
    const std::vector<tensor::Tensor>& grad_output) {
  if (grad_output.size() != 1) {
    throw std::runtime_error("MatMulFunction backward expects exactly 1 gradient");
  }

  // d(a@b)/da = grad_output @ b.T
  // d(a@b)/db = a.T @ grad_output
  std::vector<tensor::Tensor> grad_inputs;
  grad_inputs.push_back(tensor::matmul(grad_output[0], tensor::transpose(input2_, 0, 1)));
  grad_inputs.push_back(tensor::matmul(tensor::transpose(input1_, 0, 1), grad_output[0]));
  return grad_inputs;
}

// Variable implementation
Variable::Variable(const tensor::Tensor& data, bool requires_grad)
    : data_(data), requires_grad_(requires_grad), grad_fn_(nullptr) {
  if (requires_grad) {
    // Initialize gradient tensor with same shape as data but filled with zeros
    grad_ = tensor::Tensor(data.shape());
    grad_.allocate();
    // Initialize gradient to zeros (in a real implementation, we would use a zeros_like function)
    float* grad_ptr = grad_.data_ptr<float>();
    std::fill(grad_ptr, grad_ptr + grad_.numel(), 0.0f);
  }
}

Variable Variable::detach() const { return Variable(data_, false); }

// Helper class for backpropagation
class BackwardEngine {
public:
  static void execute_backward(Variable& root_var) {
    // Initialize the root gradient to ones if it's a scalar (traditional backprop starting point)
    if (root_var.shape().empty() || (root_var.numel() == 1)) {
      tensor::Tensor ones(root_var.shape());
      ones.allocate();
      float* ones_ptr = ones.data_ptr<float>();
      std::fill(ones_ptr, ones_ptr + ones.numel(), 1.0f);
      root_var.set_grad(ones);
    }

    // Build the topological ordering of the graph
    std::vector<Variable*> topo_order;
    std::unordered_set<Function*> visited_functions;

    // Start DFS from the root variable
    build_graph(root_var, topo_order, visited_functions);

    // Reverse to get proper execution order for backward pass
    std::reverse(topo_order.begin(), topo_order.end());

    // Execute backward pass in topological order
    for (Variable* var : topo_order) {
      if (!var->grad_fn())
        continue;

      auto grad_fn = var->grad_fn();
      std::vector<tensor::Tensor> grad_output = {var->grad()};
      std::vector<tensor::Tensor> grad_inputs = grad_fn->backward(grad_output);

      // Distribute gradients to input variables
      const auto& saved_vars = grad_fn->get_saved_variables();
      for (size_t i = 0; i < saved_vars.size(); ++i) {
        Variable* input_var = saved_vars[i];
        if (!input_var->requires_grad())
          continue;

        if (i < grad_inputs.size()) {
          // Accumulate gradients (for variables used multiple times)
          if (input_var->grad().data_ptr()) {
            // Assuming we have an add operation for tensors
            input_var->set_grad(tensor::add(input_var->grad(), grad_inputs[i]));
          } else {
            input_var->set_grad(grad_inputs[i]);
          }
        }
      }
    }
  }

private:
  static void build_graph(Variable& var, std::vector<Variable*>& topo_order,
                          std::unordered_set<Function*>& visited_functions) {
    if (!var.grad_fn())
      return;

    auto grad_fn = var.grad_fn();
    if (visited_functions.find(grad_fn.get()) != visited_functions.end()) {
      return;
    }

    visited_functions.insert(grad_fn.get());

    // Recursively build graph through saved variables
    for (auto input_var : grad_fn->get_saved_variables()) {
      build_graph(*input_var, topo_order, visited_functions);
    }

    topo_order.push_back(&var);
  }
};

// Variable::backward implementation
void Variable::backward() {
  if (!requires_grad_) {
    throw std::runtime_error(
        "Cannot backpropagate through a variable that doesn't require gradients");
  }

  BackwardEngine::execute_backward(*this);
}

// Operation implementations
Variable add(const Variable& a, const Variable& b) {
  // Create function object
  auto func = std::make_shared<AddFunction>();

  // Forward pass
  std::vector<tensor::Tensor> inputs = {a.data(), b.data()};
  auto outputs = func->forward(inputs);

  // Create output variable
  bool requires_grad = a.requires_grad() || b.requires_grad();
  Variable result(outputs[0], requires_grad);

  if (requires_grad) {
    // Set gradient function and save inputs for backward pass
    result.set_grad_fn(func);
    func->save_for_backward({const_cast<Variable*>(&a), const_cast<Variable*>(&b)});
  }

  return result;
}

Variable mul(const Variable& a, const Variable& b) {
  auto func = std::make_shared<MulFunction>();

  std::vector<tensor::Tensor> inputs = {a.data(), b.data()};
  auto outputs = func->forward(inputs);

  bool requires_grad = a.requires_grad() || b.requires_grad();
  Variable result(outputs[0], requires_grad);

  if (requires_grad) {
    result.set_grad_fn(func);
    func->save_for_backward({const_cast<Variable*>(&a), const_cast<Variable*>(&b)});
  }

  return result;
}

Variable matmul(const Variable& a, const Variable& b) {
  auto func = std::make_shared<MatMulFunction>();

  std::vector<tensor::Tensor> inputs = {a.data(), b.data()};
  auto outputs = func->forward(inputs);

  bool requires_grad = a.requires_grad() || b.requires_grad();
  Variable result(outputs[0], requires_grad);

  if (requires_grad) {
    result.set_grad_fn(func);
    func->save_for_backward({const_cast<Variable*>(&a), const_cast<Variable*>(&b)});
  }

  return result;
}

}  // namespace autograd
}  // namespace core
}  // namespace torchscratch