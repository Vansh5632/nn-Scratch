#pragma once

#include "core/autograd/variable.h"

namespace torchscratch {
namespace core {
namespace nn {

// Activation functions
autograd::Variable relu(const autograd::Variable& input);
autograd::Variable sigmoid(const autograd::Variable& input);
autograd::Variable tanh_activation(const autograd::Variable& input);

}  // namespace nn
}  // namespace core
}  // namespace torchscratch
