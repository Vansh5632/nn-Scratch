#include "core/optim/sgd.h"
#include <cstring>

namespace torchscratch {
namespace core {
namespace optim {

SGD::SGD(std::vector<autograd::Variable*> parameters, 
         double learning_rate, double momentum, double weight_decay)
    : parameters_(parameters), learning_rate_(learning_rate), 
      momentum_(momentum), weight_decay_(weight_decay), first_step_(true) {
    
    // Initialize velocity buffers for momentum
    if (momentum_ > 0.0) {
        velocity_.reserve(parameters_.size());
        for (auto* param : parameters_) {
            tensor::Tensor vel(param->data().shape());
            vel.allocate();
            // Initialize to zero
            float* vel_data = vel.data_ptr<float>();
            std::memset(vel_data, 0, sizeof(float) * vel.numel());
            velocity_.push_back(std::move(vel));
        }
    }
}

void SGD::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        autograd::Variable* param = parameters_[i];
        
        // Skip if no gradient
        if (param->grad().data_ptr<float>() == nullptr) {
            continue;
        }
        
        const float* grad_data = param->grad().data_ptr<float>();
        float* param_data = param->data().data_ptr<float>();
        
        // Apply weight decay if specified
        if (weight_decay_ > 0.0) {
            for (int64_t j = 0; j < param->data().numel(); ++j) {
                // grad = grad + weight_decay * param
                const_cast<float*>(grad_data)[j] += static_cast<float>(weight_decay_) * param_data[j];
            }
        }
        
        if (momentum_ > 0.0) {
            // Momentum update: v = momentum * v + grad
            float* vel_data = velocity_[i].data_ptr<float>();
            
            if (first_step_) {
                // First step: velocity = grad
                for (int64_t j = 0; j < param->data().numel(); ++j) {
                    vel_data[j] = grad_data[j];
                }
            } else {
                // Subsequent steps: velocity = momentum * velocity + grad
                for (int64_t j = 0; j < param->data().numel(); ++j) {
                    vel_data[j] = static_cast<float>(momentum_) * vel_data[j] + grad_data[j];
                }
            }
            
            // Update parameters: param = param - lr * velocity
            for (int64_t j = 0; j < param->data().numel(); ++j) {
                param_data[j] -= static_cast<float>(learning_rate_) * vel_data[j];
            }
        } else {
            // Standard SGD update: param = param - lr * grad
            for (int64_t j = 0; j < param->data().numel(); ++j) {
                param_data[j] -= static_cast<float>(learning_rate_) * grad_data[j];
            }
        }
    }
    
    first_step_ = false;
}

void SGD::zero_grad() {
    for (auto* param : parameters_) {
        if (param->grad().data_ptr<float>() != nullptr) {
            float* grad_data = param->grad().data_ptr<float>();
            std::memset(grad_data, 0, sizeof(float) * param->grad().numel());
        }
    }
}

} // namespace optim
} // namespace core
} // namespace torchscratch
