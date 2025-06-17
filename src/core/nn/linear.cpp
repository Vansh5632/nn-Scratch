#include "core/nn/linear.h"
#include "core/tensor/ops.h"
#include <random>
#include <cmath>

namespace torchscratch {
namespace core {
namespace nn {

Linear::Linear(int64_t in_features, int64_t out_features, bool bias)
    : has_bias_(bias), in_features_(in_features), out_features_(out_features) {
    
    // Create weight tensor and allocate
    tensor::Tensor weight_tensor({out_features, in_features});
    weight_tensor.allocate();
    weight_ = std::make_unique<autograd::Variable>(weight_tensor, true);
    
    if (has_bias_) {
        // Create bias tensor and allocate
        tensor::Tensor bias_tensor({out_features});
        bias_tensor.allocate();
        bias_ = std::make_unique<autograd::Variable>(bias_tensor, true);
    } else {
        // Create a dummy tensor for bias_ even when not using bias
        tensor::Tensor dummy_tensor({0});
        bias_ = std::make_unique<autograd::Variable>(dummy_tensor, false);
    }
    
    initialize_parameters();
}

autograd::Variable Linear::forward(const autograd::Variable& input) {
    // input: [batch_size, in_features]
    // weight: [out_features, in_features]
    // output: [batch_size, out_features]
    
    // Debug: Print shapes
    std::cout << "Linear::forward - input shape: [";
    for (size_t i = 0; i < input.data().shape().size(); ++i) {
        std::cout << input.data().shape()[i];
        if (i < input.data().shape().size() - 1) std::cout << ", ";
    }
    std::cout << "], dim=" << input.data().dim() << std::endl;
    
    // Compute input @ weight.T
    tensor::Tensor weight_t = tensor::transpose(weight_->data());
    std::cout << "Linear::forward - weight_t shape: [";
    for (size_t i = 0; i < weight_t.shape().size(); ++i) {
        std::cout << weight_t.shape()[i];
        if (i < weight_t.shape().size() - 1) std::cout << ", ";
    }
    std::cout << "], dim=" << weight_t.dim() << std::endl;
    
    tensor::Tensor output_tensor = tensor::matmul(input.data(), weight_t);
    std::cout << "Linear::forward - matmul result shape: [";
    for (size_t i = 0; i < output_tensor.shape().size(); ++i) {
        std::cout << output_tensor.shape()[i];
        if (i < output_tensor.shape().size() - 1) std::cout << ", ";
    }
    std::cout << "], dim=" << output_tensor.dim() << std::endl;
    
    autograd::Variable output(output_tensor, input.requires_grad() || weight_->requires_grad());
    
    if (has_bias_) {
        std::cout << "Linear::forward - bias shape: [";
        for (size_t i = 0; i < bias_->data().shape().size(); ++i) {
            std::cout << bias_->data().shape()[i];
            if (i < bias_->data().shape().size() - 1) std::cout << ", ";
        }
        std::cout << "], dim=" << bias_->data().dim() << std::endl;
        
        // Add bias (broadcasting)
        tensor::Tensor result_tensor = tensor::add(output.data(), bias_->data());
        output = autograd::Variable(result_tensor, output.requires_grad() || bias_->requires_grad());
    }
    
    return output;
}

std::vector<autograd::Variable*> Linear::parameters() {
    std::vector<autograd::Variable*> params;
    params.push_back(weight_.get());
    if (has_bias_) {
        params.push_back(bias_.get());
    }
    return params;
}

void Linear::zero_grad() {
    if (weight_->grad().data_ptr<float>() != nullptr) {
        float* weight_grad_data = weight_->grad().data_ptr<float>();
        for (int64_t i = 0; i < weight_->grad().numel(); ++i) {
            weight_grad_data[i] = 0.0f;
        }
    }
    
    if (has_bias_ && bias_->grad().data_ptr<float>() != nullptr) {
        float* bias_grad_data = bias_->grad().data_ptr<float>();
        for (int64_t i = 0; i < bias_->grad().numel(); ++i) {
            bias_grad_data[i] = 0.0f;
        }
    }
}

void Linear::initialize_parameters() {
    // Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    float bound = std::sqrt(6.0f / (in_features_ + out_features_));
    std::uniform_real_distribution<float> dis(-bound, bound);
    
    // Initialize weight
    float* weight_data = weight_->data().data_ptr<float>();
    for (int64_t i = 0; i < weight_->data().numel(); ++i) {
        weight_data[i] = dis(gen);
    }
    
    // Initialize bias to zero
    if (has_bias_) {
        float* bias_data = bias_->data().data_ptr<float>();
        for (int64_t i = 0; i < bias_->data().numel(); ++i) {
            bias_data[i] = 0.0f;
        }
    }
}

} // namespace nn
} // namespace core
} // namespace torchscratch
