# Contributing to nn-from-Scratch

Thank you for considering contributing to PyTorch-from-Scratch! This document outlines the process for contributing to this project and helps you get started.

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Development Environment Setup](#development-environment-setup)
- [Development Workflow](#development-workflow)
  - [Building the Project](#building-the-project)
  - [Testing](#testing)
  - [Code Quality Tools](#code-quality-tools)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Project Structure](#project-structure)
- [License](#license)

## Project Overview

PyTorch-from-Scratch is a minimal deep learning framework built from the ground up to demystify the internals of modern machine learning libraries like PyTorch and TensorFlow. The project includes both a C++ core library and Python bindings to provide a PyTorch-like API.

## Getting Started

### Prerequisites

To contribute to this project, you'll need:

- **C++ Compiler** (GCC 7+ or Clang 5+)
- **CMake** (3.14+)
- **Python** (3.7+)
- **Git**

Additional tools:
- **clang-format** (for code formatting)
- **clang-tidy** (for static analysis)
- **cppcheck** (for additional static analysis)

### Development Environment Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/pytorch-from-scratch.git
cd pytorch-from-scratch
```

2. **Install dependencies**

On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev python3-pip clang-format clang-tidy cppcheck
pip3 install -r requirements-dev.txt
```

## Development Workflow

### Building the Project

We provide a helper script to build the project with common configurations:

```bash
# One-command build
./scripts/build.sh
```

Or manually with CMake:

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Testing

To run C++ tests:

```bash
# From the build directory
ctest --verbose
# Or directly
./build/test_tensor
```

To run Python tests:

```bash
python -m pytest python/tests
```

### All-in-One Development Script

For convenience, we've provided a script to handle building, testing, and code quality checks:

```bash
./scripts/dev.sh
```

### Code Quality Tools

Format your code:

```bash
# Format C++ code
make format
# Or directly
./scripts/format_cpp.sh
```

Run linters:

```bash
# Lint C++ code
make lint
# Or directly
./scripts/lint_cpp.sh
```

## Pull Request Process

1. **Fork the repository** on GitHub
2. **Create a new branch** for your feature or bugfix
3. **Commit your changes** with clear commit messages
4. **Run tests and linters** to ensure code quality
5. **Submit a pull request** against the `main` branch
6. **Update the documentation** as needed

## Coding Standards

- **C++**: Follow the project's style guide as enforced by clang-format
  - Use C++14 features
  - Include documentation for public APIs
  - Follow the Google C++ Style Guide with project-specific modifications
  
- **Python**: Follow PEP 8 with these guidelines:
  - Use 4 spaces for indentation
  - Maximum line length of 88 characters (Black formatter)
  - Docstrings in Google style

## Project Structure

```
pytorch-from-scratch/
├── cmake/                     # CMake configuration files
├── include/                   # Public C++ headers
│   └── core/
│       ├── tensor/           # Tensor core implementation
│       ├── autograd/         # Autograd components
│       └── nn/               # Neural network components
├── src/                       # C++ implementation
│   ├── core/
│   │   ├── tensor/
│   │   ├── autograd/
│   │   └── nn/
│   └── python/               # Python binding sources
├── python/                    # Python package
│   └── torchscratch/
├── test/                      # Comprehensive tests
│   ├── cpp/
│   └── python/
├── scripts/                   # Maintenance scripts
├── examples/                  # Example projects
└── docs/                      # Documentation
```

## License

By contributing to PyTorch-from-Scratch, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
```
