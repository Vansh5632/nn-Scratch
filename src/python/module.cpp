#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations of initialization functions
void init_tensor(py::module& m);
void init_nn(py::module& m);
void init_optim(py::module& m);

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