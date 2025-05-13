#!/bin/bash
# filepath: /home/vansh5632/sudo/nn/scripts/build.sh

set -e

# ANSI color codes
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m"  # No Color

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Parse arguments
BUILD_TYPE="Debug"
CLEAN=0
JOBS=$(nproc)

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --release) BUILD_TYPE="Release" ;;
        --debug) BUILD_TYPE="Debug" ;;
        --clean) CLEAN=1 ;;
        -j*) JOBS="${1:2}" ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --release   Build in release mode"
            echo "  --debug     Build in debug mode (default)"
            echo "  --clean     Clean build directory before building"
            echo "  -j<N>       Number of parallel jobs (default: nproc)"
            echo "  --help      Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Build directory
BUILD_DIR="$PROJECT_ROOT/build"

# Clean if requested
if [ "$CLEAN" -eq 1 ]; then
    echo -e "${YELLOW}Cleaning build directory: $BUILD_DIR${NC}"
    rm -rf "$BUILD_DIR"
fi

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo -e "${GREEN}Configuring with CMake - Build type: $BUILD_TYPE${NC}"
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..

# Build
echo -e "${GREEN}Building with $JOBS jobs...${NC}"
cmake --build . -- -j"$JOBS"

echo -e "${GREEN}Build complete! 🚀${NC}"