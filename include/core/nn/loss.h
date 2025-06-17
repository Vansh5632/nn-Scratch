#pragma once

#include "core/autograd/variable.h"

namespace torchscratch {
namespace core {
namespace nn {

// Loss functions
autograd::Variable mse_loss(const autograd::Variable& predicted, 
                           const autograd::Variable& target);

autograd::Variable cross_entropy_loss(const autograd::Variable& predicted,
                                     const autograd::Variable& target);

autograd::Variable binary_cross_entropy_loss(const autograd::Variable& predicted,
                                            const autograd::Variable& target);

} // namespace nn
} // namespace core
} // namespace torchscratch
