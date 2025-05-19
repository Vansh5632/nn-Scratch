# Contributing to TorchScratch

Thank you for your interest in contributing to TorchScratch! This document provides guidelines and instructions to help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Development Environment](#development-environment)
  - [Building the Project](#building-the-project)
  - [Running Tests](#running-tests)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Pull Requests](#pull-requests)
- [Development Guidelines](#development-guidelines)
  - [C++ Style Guide](#c-style-guide)
  - [Python Style Guide](#python-style-guide)
  - [Documentation](#documentation)
  - [Testing](#testing)
- [Project Structure](#project-structure)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone. Please be kind, patient, and considerate of others.

## Getting Started

### Development Environment

#### Prerequisites

- CMake (3.17 or higher)
- C++ compiler with C++14 support (GCC 7+, Clang 5+, MSVC 19.14+)
- Python 3.7 or higher
- NumPy

#### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/torchscratch.git
   cd torchscratch
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

### Building the Project

#### Option 1: Using the build script

The quickest way to build the project is to use the provided build script:

```bash
./scripts/build.sh
```

#### Option 2: Manual build

```bash
mkdir -p build
cd build
cmake ..
make
```

#### Option 3: Development installation

For Python development, you can install the package in development mode:

```bash
pip install -e .
```

### Running Tests

#### C++ Tests

```bash
cd build
ctest
```

Or run specific test executables:

```bash
./test_tensor
./test_autograd
```

#### Python Tests

```bash
pytest
```

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment information (OS, compiler version, Python version, etc.)

### Suggesting Features

Feature requests are welcome! Please include:

- A clear description of the feature
- The motivation and use case for the feature
- Any implementation ideas you may have

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make your changes**
4. **Run tests and linting**: Make sure your changes pass all existing tests and follow the code style
5. **Commit your changes**: Use meaningful commit messages
6. **Push to your fork**: `git push origin feature/my-feature`
7. **Create a pull request**: Provide a clear description of your changes

## Development Guidelines

### C++ Style Guide

- Use C++14 features
- Follow the Google C++ style guide with some modifications
- Run clang-format before committing: `./scripts/format_cpp.sh`
- Use `snake_case` for variable and function names
- Use `PascalCase` for class names

### Python Style Guide

- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 88 characters (compatible with black formatter)
- Use docstrings for all public functions, classes, and methods

### Documentation

- Document all public APIs
- Include examples where appropriate
- Update the documentation when modifying code

### Testing

- Write tests for all new features
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage

## Project Structure

Please see the [README.md](README.md) for an overview of the project structure.

## Thank You!

Your contributions help make TorchScratch better. We appreciate your time and effort!
