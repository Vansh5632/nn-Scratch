#pragma once

#include "core/autograd/variable.h"
#include <vector>
#include <memory>

namespace torchscratch {
namespace core {
namespace nn {

class Linear {
public:
    Linear(int64_t in_features, int64_t out_features, bool bias = true);
    
    autograd::Variable forward(const autograd::Variable& input);
    std::vector<autograd::Variable*> parameters();
    
    void zero_grad();
    
    // Getters
    int64_t in_features() const { return in_features_; }
    int64_t out_features() const { return out_features_; }
    bool has_bias() const { return has_bias_; }
    
    // Access to parameters
    autograd::Variable& weight() { return *weight_; }
    autograd::Variable& bias() { return *bias_; }
    const autograd::Variable& weight() const { return *weight_; }
    const autograd::Variable& bias() const { return *bias_; }
    
private:
    bool has_bias_;
    int64_t in_features_;
    int64_t out_features_;
    std::unique_ptr<autograd::Variable> weight_;
    std::unique_ptr<autograd::Variable> bias_;
    
    void initialize_parameters();
};

} // namespace nn
} // namespace core
} // namespace torchscratch
