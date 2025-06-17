#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/optim/sgd.h"

namespace py = pybind11;
namespace ts = torchscratch;

void init_optim(py::module& m) {
  // Create optimizer submodule
  auto optim = m.def_submodule("optim", "Optimization algorithms");

  // SGD optimizer
  py::class_<ts::core::optim::SGD>(optim, "SGD")
      .def(py::init<std::vector<ts::core::autograd::Variable*>, double, double, double>(),
           py::arg("parameters"), py::arg("lr"), py::arg("momentum") = 0.0,
           py::arg("weight_decay") = 0.0)
      .def("step", &ts::core::optim::SGD::step, "Perform a single optimization step")
      .def("zero_grad", &ts::core::optim::SGD::zero_grad, "Zero out gradients")
      .def("learning_rate", &ts::core::optim::SGD::learning_rate)
      .def("momentum", &ts::core::optim::SGD::momentum)
      .def("weight_decay", &ts::core::optim::SGD::weight_decay)
      .def("set_learning_rate", &ts::core::optim::SGD::set_learning_rate);
}
