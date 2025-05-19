#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations of initialization functions
void init_tensor(py::module& m);

// Future module initializations
void init_nn(py::module& m) {
  // Create neural network submodule
  auto nn = m.def_submodule("nn", "Neural network module");

  // This is a placeholder for future NN components
  // We'll add Linear, Conv2D, ReLU classes, etc.
}

void init_optim(py::module& m) {
  // Create optimizer submodule
  auto optim = m.def_submodule("optim", "Optimization algorithms");

  // This is a placeholder for future optimizers
  // We'll add SGD, Adam, etc.
}

PYBIND11_MODULE(torchscratch_cpp, m) {
  m.doc() = "TorchScratch: A neural network library built from scratch";

  // Initialize tensor module
  init_tensor(m);

  // Initialize neural network module
  init_nn(m);

  // Initialize optimizer module
  init_optim(m);

  // Version info
  m.attr("__version__") = "0.1.0";
}