#include "core/tensor/tensor.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

#include "core/autograd/function.h"
#include "core/autograd/variable.h"
#include "core/tensor/ops.h"

namespace py = pybind11;
namespace ts = torchscratch;

// Helper function to convert numpy array to our Tensor
ts::core::tensor::Tensor numpy_to_tensor(py::array_t<float> array) {
  py::buffer_info buf = array.request();
  std::vector<int64_t> shape;
  for (size_t i = 0; i < static_cast<size_t>(buf.ndim); i++) {
    shape.push_back(static_cast<int64_t>(buf.shape[i]));
  }

  ts::core::tensor::Tensor tensor(shape);
  tensor.allocate();

  // Copy data from numpy array to our tensor
  float* tensor_data = tensor.data_ptr<float>();
  std::memcpy(tensor_data, buf.ptr, sizeof(float) * static_cast<size_t>(tensor.numel()));

  return tensor;
}

// Helper function to convert our Tensor to numpy array
py::array_t<float> tensor_to_numpy(const ts::core::tensor::Tensor& tensor) {
  // Get shape for numpy array
  std::vector<int64_t> shape = tensor.shape();
  std::vector<ssize_t> numpy_shape(shape.begin(), shape.end());

  // Create numpy array
  py::array_t<float> array(numpy_shape);
  py::buffer_info buf = array.request();

  // Copy data from tensor to numpy array
  std::memcpy(buf.ptr, tensor.data_ptr<float>(),
              sizeof(float) * static_cast<size_t>(tensor.numel()));

  return array;
}

void init_tensor(py::module& m) {
  // Define Tensor class
  py::class_<ts::core::tensor::Tensor>(m, "Tensor")
      .def(py::init<>())
      .def(py::init<const std::vector<int64_t>&>())
      .def(py::init([](py::array_t<float> array) { return numpy_to_tensor(array); }))
      .def("shape", &ts::core::tensor::Tensor::shape)
      .def("strides", &ts::core::tensor::Tensor::strides)
      .def("dim", &ts::core::tensor::Tensor::dim)
      .def("numel", &ts::core::tensor::Tensor::numel)
      .def("is_contiguous", &ts::core::tensor::Tensor::is_contiguous)
      .def("reshape", &ts::core::tensor::Tensor::reshape)
      .def("clone", &ts::core::tensor::Tensor::clone)
      .def("allocate", &ts::core::tensor::Tensor::allocate)
      .def("deallocate", &ts::core::tensor::Tensor::deallocate)
      .def("is_cuda", &ts::core::tensor::Tensor::is_cuda)
      .def("numpy", [](const ts::core::tensor::Tensor& tensor) { return tensor_to_numpy(tensor); })
      .def("__repr__", [](const ts::core::tensor::Tensor& tensor) {
        std::stringstream ss;
        ss << "Tensor(shape=[";
        const auto& shape = tensor.shape();
        for (size_t i = 0; i < shape.size(); ++i) {
          ss << shape[i];
          if (i < shape.size() - 1)
            ss << ", ";
        }
        ss << "])";
        return ss.str();
      });

  // Tensor operations
  m.def("add", &ts::core::tensor::add, "Element-wise addition of two tensors");
  m.def("mul", &ts::core::tensor::mul, "Element-wise multiplication of two tensors");
  m.def("matmul", &ts::core::tensor::matmul, "Matrix multiplication");
  m.def("transpose", &ts::core::tensor::transpose, "Transpose a tensor", py::arg("tensor"),
        py::arg("dim0") = 0, py::arg("dim1") = 1);

  // Define Variable class for autograd
  py::class_<ts::core::autograd::Variable>(m, "Variable")
      .def(py::init<const ts::core::tensor::Tensor&, bool>(), py::arg("tensor"),
           py::arg("requires_grad") = false)
      .def("data", &ts::core::autograd::Variable::data, py::return_value_policy::reference)
      .def("grad", &ts::core::autograd::Variable::grad, py::return_value_policy::reference)
      .def("requires_grad", &ts::core::autograd::Variable::requires_grad)
      .def("backward", &ts::core::autograd::Variable::backward)
      .def("detach", &ts::core::autograd::Variable::detach)
      .def("__repr__", [](const ts::core::autograd::Variable& var) {
        std::stringstream ss;
        ss << "Variable(";
        ss << "requires_grad=" << (var.requires_grad() ? "True" : "False");
        ss << ")";
        return ss.str();
      });

  // Variable operations for autograd
  m.def(
      "add",
      [](const ts::core::autograd::Variable& a, const ts::core::autograd::Variable& b) {
        // Create an AddFunction to handle backward pass
        auto func = std::make_shared<ts::core::autograd::AddFunction>();

        // Forward pass
        ts::core::tensor::Tensor result_tensor = ts::core::tensor::add(a.data(), b.data());

        // Create result variable with requires_grad if either input requires grad
        bool requires_grad = a.requires_grad() || b.requires_grad();
        ts::core::autograd::Variable result(result_tensor, requires_grad);

        // Setup backward computation if needed
        if (requires_grad) {
          result.set_grad_fn(func);
          // Save inputs for backward pass
          func->save_for_backward({const_cast<ts::core::autograd::Variable*>(&a),
                                   const_cast<ts::core::autograd::Variable*>(&b)});
        }

        return result;
      },
      "Add two Variables");

  m.def(
      "mul",
      [](const ts::core::autograd::Variable& a, const ts::core::autograd::Variable& b) {
        auto func = std::make_shared<ts::core::autograd::MulFunction>();

        // Forward pass
        ts::core::tensor::Tensor result_tensor = ts::core::tensor::mul(a.data(), b.data());

        bool requires_grad = a.requires_grad() || b.requires_grad();
        ts::core::autograd::Variable result(result_tensor, requires_grad);

        if (requires_grad) {
          result.set_grad_fn(func);
          func->save_for_backward({const_cast<ts::core::autograd::Variable*>(&a),
                                   const_cast<ts::core::autograd::Variable*>(&b)});
        }

        return result;
      },
      "Multiply two Variables");

  m.def(
      "matmul",
      [](const ts::core::autograd::Variable& a, const ts::core::autograd::Variable& b) {
        auto func = std::make_shared<ts::core::autograd::MatMulFunction>();

        // Forward pass
        ts::core::tensor::Tensor result_tensor = ts::core::tensor::matmul(a.data(), b.data());

        bool requires_grad = a.requires_grad() || b.requires_grad();
        ts::core::autograd::Variable result(result_tensor, requires_grad);

        if (requires_grad) {
          result.set_grad_fn(func);
          func->save_for_backward({const_cast<ts::core::autograd::Variable*>(&a),
                                   const_cast<ts::core::autograd::Variable*>(&b)});
        }

        return result;
      },
      "Matrix multiplication of two Variables");
}