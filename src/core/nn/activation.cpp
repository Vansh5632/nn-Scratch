#include "core/nn/activation.h"
#include "core/autograd/function.h"
#include <cmath>
#include <algorithm>

namespace torchscratch {
namespace core {
namespace nn {

// ReLU Forward Function
class ReLUFunction : public autograd::Function {
public:
    std::vector<tensor::Tensor> forward(const std::vector<tensor::Tensor>& inputs) override {
        const tensor::Tensor& input = inputs[0];
        tensor::Tensor output(input.shape());
        output.allocate();
        
        const float* input_data = input.data_ptr<float>();
        float* output_data = output.data_ptr<float>();
        
        for (int64_t i = 0; i < input.numel(); ++i) {
            output_data[i] = std::max(0.0f, input_data[i]);
        }
        
        return {output};
    }
    
    std::vector<tensor::Tensor> backward(const std::vector<tensor::Tensor>& grad_output) override {
        const auto& saved_vars = get_saved_variables();
        const tensor::Tensor& input = saved_vars[0]->data();
        const tensor::Tensor& grad_out = grad_output[0];
        tensor::Tensor grad_input(input.shape());
        grad_input.allocate();
        
        const float* input_data = input.data_ptr<float>();
        const float* grad_output_data = grad_out.data_ptr<float>();
        float* grad_input_data = grad_input.data_ptr<float>();
        
        for (int64_t i = 0; i < input.numel(); ++i) {
            grad_input_data[i] = input_data[i] > 0.0f ? grad_output_data[i] : 0.0f;
        }
        
        return {grad_input};
    }
    
    std::string name() const override { return "ReLUFunction"; }
};

// Sigmoid Forward Function
class SigmoidFunction : public autograd::Function {
public:
    std::vector<tensor::Tensor> forward(const std::vector<tensor::Tensor>& inputs) override {
        const tensor::Tensor& input = inputs[0];
        tensor::Tensor output(input.shape());
        output.allocate();
        
        const float* input_data = input.data_ptr<float>();
        float* output_data = output.data_ptr<float>();
        
        for (int64_t i = 0; i < input.numel(); ++i) {
            output_data[i] = 1.0f / (1.0f + std::exp(-input_data[i]));
        }
        
        return {output};
    }
    
    std::vector<tensor::Tensor> backward(const std::vector<tensor::Tensor>& grad_output) override {
        const auto& saved_vars = get_saved_variables();
        const tensor::Tensor& output = saved_vars[0]->data(); // We save the output for sigmoid
        const tensor::Tensor& grad_out = grad_output[0];
        tensor::Tensor grad_input(output.shape());
        grad_input.allocate();
        
        const float* output_data = output.data_ptr<float>();
        const float* grad_output_data = grad_out.data_ptr<float>();
        float* grad_input_data = grad_input.data_ptr<float>();
        
        for (int64_t i = 0; i < output.numel(); ++i) {
            float sig = output_data[i];
            grad_input_data[i] = grad_output_data[i] * sig * (1.0f - sig);
        }
        
        return {grad_input};
    }
    
    std::string name() const override { return "SigmoidFunction"; }
};

// Tanh Forward Function
class TanhFunction : public autograd::Function {
public:
    std::vector<tensor::Tensor> forward(const std::vector<tensor::Tensor>& inputs) override {
        const tensor::Tensor& input = inputs[0];
        tensor::Tensor output(input.shape());
        output.allocate();
        
        const float* input_data = input.data_ptr<float>();
        float* output_data = output.data_ptr<float>();
        
        for (int64_t i = 0; i < input.numel(); ++i) {
            output_data[i] = std::tanh(input_data[i]);
        }
        
        return {output};
    }
    
    std::vector<tensor::Tensor> backward(const std::vector<tensor::Tensor>& grad_output) override {
        const auto& saved_vars = get_saved_variables();
        const tensor::Tensor& output = saved_vars[0]->data(); // We save the output for tanh
        const tensor::Tensor& grad_out = grad_output[0];
        tensor::Tensor grad_input(output.shape());
        grad_input.allocate();
        
        const float* output_data = output.data_ptr<float>();
        const float* grad_output_data = grad_out.data_ptr<float>();
        float* grad_input_data = grad_input.data_ptr<float>();
        
        for (int64_t i = 0; i < output.numel(); ++i) {
            float tanh_val = output_data[i];
            grad_input_data[i] = grad_output_data[i] * (1.0f - tanh_val * tanh_val);
        }
        
        return {grad_input};
    }
    
    std::string name() const override { return "TanhFunction"; }
};

autograd::Variable relu(const autograd::Variable& input) {
    auto func = std::make_shared<ReLUFunction>();
    
    // Forward pass
    std::vector<tensor::Tensor> result_tensors = func->forward({input.data()});
    tensor::Tensor result_tensor = result_tensors[0];
    
    bool requires_grad = input.requires_grad();
    autograd::Variable result(result_tensor, requires_grad);
    
    if (requires_grad) {
        result.set_grad_fn(func);
        // Create a temporary Variable for saving
        auto input_var = const_cast<autograd::Variable*>(&input);
        func->save_for_backward({input_var}); // Save input for backward
    }
    
    return result;
}

autograd::Variable sigmoid(const autograd::Variable& input) {
    auto func = std::make_shared<SigmoidFunction>();
    
    // Forward pass
    std::vector<tensor::Tensor> result_tensors = func->forward({input.data()});
    tensor::Tensor result_tensor = result_tensors[0];
    
    bool requires_grad = input.requires_grad();
    autograd::Variable result(result_tensor, requires_grad);
    
    if (requires_grad) {
        result.set_grad_fn(func);
        // Save output for sigmoid backward pass
        auto result_var = new autograd::Variable(result_tensor, false);
        func->save_for_backward({result_var}); // Save output for sigmoid
    }
    
    return result;
}

autograd::Variable tanh_activation(const autograd::Variable& input) {
    auto func = std::make_shared<TanhFunction>();
    
    // Forward pass  
    std::vector<tensor::Tensor> result_tensors = func->forward({input.data()});
    tensor::Tensor result_tensor = result_tensors[0];
    
    bool requires_grad = input.requires_grad();
    autograd::Variable result(result_tensor, requires_grad);
    
    if (requires_grad) {
        result.set_grad_fn(func);
        // Save output for tanh backward pass
        auto result_var = new autograd::Variable(result_tensor, false);
        func->save_for_backward({result_var}); // Save output for tanh
    }
    
    return result;
}

} // namespace nn
} // namespace core
} // namespace torchscratch
