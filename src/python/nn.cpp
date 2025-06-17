#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/nn/linear.h"
#include "core/nn/activation.h"
#include "core/nn/loss.h"

namespace py = pybind11;
namespace ts = torchscratch;

void init_nn(py::module& m) {
    // Create neural network submodule
    auto nn = m.def_submodule("nn", "Neural network module");
    
    // Linear layer
    py::class_<ts::core::nn::Linear>(nn, "Linear")
        .def(py::init<int64_t, int64_t, bool>(),
             py::arg("in_features"), py::arg("out_features"), py::arg("bias") = true)
        .def("forward", &ts::core::nn::Linear::forward)
        .def("__call__", &ts::core::nn::Linear::forward)
        .def("parameters", &ts::core::nn::Linear::parameters,
             py::return_value_policy::reference)
        .def("zero_grad", &ts::core::nn::Linear::zero_grad)
        .def("in_features", &ts::core::nn::Linear::in_features)
        .def("out_features", &ts::core::nn::Linear::out_features)
        .def("has_bias", &ts::core::nn::Linear::has_bias)
        .def("weight", static_cast<ts::core::autograd::Variable& (ts::core::nn::Linear::*)()>(&ts::core::nn::Linear::weight),
             py::return_value_policy::reference)
        .def("bias", static_cast<ts::core::autograd::Variable& (ts::core::nn::Linear::*)()>(&ts::core::nn::Linear::bias),
             py::return_value_policy::reference);
    
    // Activation functions
    nn.def("relu", &ts::core::nn::relu, "ReLU activation function");
    nn.def("sigmoid", &ts::core::nn::sigmoid, "Sigmoid activation function");
    nn.def("tanh", &ts::core::nn::tanh_activation, "Tanh activation function");
    
    // Loss functions
    nn.def("mse_loss", &ts::core::nn::mse_loss, "Mean Squared Error loss");
    nn.def("binary_cross_entropy_loss", &ts::core::nn::binary_cross_entropy_loss, "Binary Cross Entropy loss");
    nn.def("cross_entropy_loss", &ts::core::nn::cross_entropy_loss, "Cross Entropy loss");
}
