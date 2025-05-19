#!/bin/bash
# Development environment setup script for TorchScratch
# This script sets up a development environment for working on TorchScratch.

set -e  # Exit on error

VENV_DIR="venv"
PYTHON="python3"

echo "=== TorchScratch Development Environment Setup ==="

# Check for Python
if ! command -v $PYTHON &> /dev/null; then
    echo "Error: $PYTHON not found. Please install Python 3.7 or newer."
    exit 1
fi

# Check Python version
PY_VERSION=$($PYTHON --version 2>&1 | cut -d" " -f2)
PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 7 ]); then
    echo "Error: Python 3.7+ is required (found $PY_VERSION)"
    exit 1
fi

echo "Using Python $PY_VERSION"

# Check for required tools
echo "Checking for required tools..."

if ! command -v cmake &> /dev/null; then
    echo "Error: cmake not found. Please install CMake 3.17 or newer."
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | cut -d" " -f3)
echo "Found CMake $CMAKE_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    $PYTHON -m venv $VENV_DIR
else
    echo "Virtual environment already exists in $VENV_DIR."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Install development dependencies
echo "Installing development dependencies..."
pip install -U pip setuptools wheel
pip install numpy pytest pytest-cov black isort flake8

# Install the project in development mode
echo "Installing TorchScratch in development mode..."
pip install -e .

# Build the project
echo "Building the C++ components..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..

echo ""
echo "=== Development environment setup complete! ==="
echo "You can activate the virtual environment with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To build the project:"
echo "  cd build && make"
echo ""
echo "To run tests:"
echo "  cd build && ctest  # For C++ tests"
echo "  pytest             # For Python tests"
echo ""
echo "Happy coding!"