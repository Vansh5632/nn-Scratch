#pragma once

#include "core/autograd/variable.h"
#include <vector>

namespace torchscratch {
namespace core {
namespace optim {

class SGD {
public:
    SGD(std::vector<autograd::Variable*> parameters, 
        double learning_rate, double momentum = 0.0, double weight_decay = 0.0);
    
    void step();
    void zero_grad();
    
    // Getters
    double learning_rate() const { return learning_rate_; }
    double momentum() const { return momentum_; }
    double weight_decay() const { return weight_decay_; }
    
    // Setters
    void set_learning_rate(double lr) { learning_rate_ = lr; }
    
private:
    std::vector<autograd::Variable*> parameters_;
    double learning_rate_;
    double momentum_;
    double weight_decay_;
    std::vector<tensor::Tensor> velocity_;
    bool first_step_;
};

} // namespace optim
} // namespace core
} // namespace torchscratch
