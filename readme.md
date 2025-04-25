# 🧠 PyTorch-from-Scratch

**PyTorch-from-Scratch** is a minimal deep learning framework built from the ground up to **demystify the internals** of modern machine learning libraries like PyTorch and TensorFlow.  
It serves as a learning tool, a research sandbox, and a hackable base for experiments.

---

## 🚀 Project Vision

- **Educational First**: Learn how tensor operations, autograd, and neural networks work under the hood  
- **Hackable and Modular**: Easily extend core functionality in Python or C++  
- **Lightweight by Design**: Minimal dependencies, designed to run on CPU with optional CUDA support

---

## 🧱 Core Components

### 1. Tensor System
Multi-dimensional array library with core tensor operations and shape handling.

- Memory management with support for CPU (GPU support via CUDA planned)
- Math ops: `add`, `mul`, `matmul`, etc.
- Broadcasting and shape inference

### 2. Autograd Engine
Dynamic computation graph with automatic differentiation.

- Forward graph construction on-the-fly
- Reverse-mode autodiff (`.backward()`)
- Custom backward functions supported

### 3. Python API
Python-first design that mimics PyTorch's style for user familiarity.

- Operator overloading (e.g., `a + b`)
- Interoperable with NumPy
- Clean class-based architecture

### 4. Neural Network Module
Composable building blocks for model construction.

- Layers like `Linear`, `Conv2D`, `ReLU`
- Parameter tracking and weight initialization
- Module nesting and hooks

### 5. Optimization System
Standard training optimizers for parameter updates.

- Support for `SGD`, `Adam`, etc.
- Learning rate scheduling
- Momentum and weight decay

---

## 🔧 Architecture Flow

```python
x = Tensor(...)  
y = model(x)  
loss = y.sum()  
loss.backward()  
```

Under the hood:

- Python objects map to efficient C++ Tensor structures  
- Operations build a computation graph dynamically  
- Gradients flow back automatically through the graph  
- Optimizer steps update weights  

---

## ⚙️ Technical Architecture

### Project Layers

```
[ Python API ]
     ↓
[ pybind11 Bindings ]
     ↓
[ C++ Core Library ]
```

### Build Flow

- Core C++: `libpytorch_core.so` (tensor ops, autograd)  
- Python Bindings: via **pybind11**  
- Python Package: high-level modules (`nn`, `optim`, etc.)

---

## 📦 Project Structure

```
nn-from-scratch/
├── cmake/                     # CMake configuration files
├── include/                   # Public C++ headers
│   ├── core/
│   │   ├── tensor/           # Tensor core implementation
│   │   │   ├── tensor.h
│   │   │   ├── tensor_impl.h
│   │   │   └── tensor_ops.h
│   │   ├── autograd/         # Autograd components
│   │   │   ├── function.h
│   │   │   ├── engine.h
│   │   │   └── variable.h
│   │   ├── nn/               # Neural network components
│   │   │   ├── modules.h
│   │   │   └── functional.h
│   │   └── utils/            # Utilities
│   │       ├── logging.h
│   │       └── serializer.h
├── src/                       # C++ implementation
│   ├── core/
│   │   ├── tensor/
│   │   │   ├── tensor.cpp
│   │   │   └── tensor_ops.cpp
│   │   ├── autograd/
│   │   │   ├── engine.cpp
│   │   │   └── function.cpp
│   │   └── nn/
│   │       └── modules.cpp
│   └── python/               # Python binding sources
│       ├── module.cpp
│       └── tensor.cpp
├── python/                    # Python package
│   ├── torchscratch/
│   │   ├── __init__.py
│   │   ├── tensor.py         # Python tensor class
│   │   ├── nn/               # Neural network modules
│   │   │   ├── __init__.py
│   │   │   ├── linear.py
│   │   │   └── functional.py
│   │   ├── optim/            # Optimizers
│   │   │   ├── __init__.py
│   │   │   └── sgd.py
│   │   ├── utils/            # Python utilities
│   │   │   ├── data.py
│   │   │   └── logger.py
│   │   └── csrc/             # Compiled extensions
├── test/                      # Comprehensive tests
│   ├── cpp/                  # C++ tests
│   │   ├── tensor/
│   │   │   └── test_tensor.cpp
│   │   └── autograd/
│   │       └── test_autograd.cpp
│   └── python/               # Python tests
│       ├── test_tensor.py
│       └── test_nn.py
├── third_party/              # External dependencies
│   ├── pybind11/             # Python binding library
│   └── googletest/           # Google Test framework
├── examples/                 # Example projects
│   ├── mnist/
│   └── cifar10/
├── docs/                     # Documentation
│   ├── source/
│   │   ├── getting_started.rst
│   │   └── api_reference.rst
│   └── Makefile
├── scripts/                  # Maintenance scripts
│   ├── build.sh
│   ├── format_code.sh
│   └── run_tests.sh
├── .github/
│   └── workflows/           # CI/CD pipelines
│       ├── build.yml
│       └── test.yml
├── CMakeLists.txt           # Root CMake configuration
├── setup.py                 # Python package installation
├── pyproject.toml          # Modern Python packaging config
├── LICENSE
├── CONTRIBUTING.md
├── README.md
└── .gitignore
```

---

## 🚰 Key Challenges & Goals

| Area         | Challenge                           | Solution Goal                                |
|--------------|-------------------------------------|----------------------------------------------|
| Memory       | Tensor allocation & reuse           | Smart allocator & GPU memory hooks (CUDA)    |
| Performance  | Python-C++ overhead                 | Minimize bindings + vectorized kernels       |
| Autograd     | In-place ops, non-diff ops          | Robust backward graph & error handling       |
| API Design   | PyTorch-like UX vs complexity       | Intuitive syntax + clean abstraction layers  |

---

## 🌐 Future Plans

- ✅ CPU-first stable backend  
- 🛠️ CUDA integration (GPU acceleration)  
- 🔄 ONNX export & model serialization  
- 🌍 Distributed training (multi-GPU, autograd support)  
- 📦 Plugin system for custom ops/layers/optimizers  

---

## 📚 Why This Project Matters

| Role         | Value                                                              |
|--------------|--------------------------------------------------------------------|
| Learners     | Understand *how* PyTorch works under the hood                     |
| Researchers  | Safely prototype new models, backprop ideas, and layers           |
| Developers   | Contribute to a clean, modular, and understandable framework      |

---

## ✅ What Success Looks Like

Users can:

- Define custom neural networks  
- Train with `.backward()` + optimizers  
- Extend components in both C++ and Python  

The codebase stays:

- 📦 **Lightweight** – minimal setup  
- 🧺 **Testable** – isolated units + integration tests  
- 💡 **Hackable** – core logic easy to follow and modify

---

## 🤝 Contributing

Pull requests are welcome! We aim for clean code, thoughtful abstractions, and a welcoming space for learning and experimentation.

---

## 📄 License

MIT License – Free to use, learn from, and build on.

