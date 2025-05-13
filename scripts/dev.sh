#!/bin/bash
# filepath: /home/vansh5632/sudo/nn/scripts/dev.sh

set -e  # Exit immediately if a command exits with a non-zero status

# ANSI color codes
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m"  # No Color

# Working directory to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

CLEAN_BUILD=0
SKIP_BUILD=0
SKIP_TEST=0
SKIP_LINT=0
SKIP_FORMAT=0
BUILD_TYPE="Debug"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --clean) CLEAN_BUILD=1 ;;
        --no-build) SKIP_BUILD=1 ;;
        --no-test) SKIP_TEST=1 ;;
        --no-lint) SKIP_LINT=1 ;;
        --no-format) SKIP_FORMAT=1 ;;
        --release) BUILD_TYPE="Release" ;;
        --debug) BUILD_TYPE="Debug" ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --clean      Clean the build directory before building"
            echo "  --no-build   Skip building the project"
            echo "  --no-test    Skip running tests"
            echo "  --no-lint    Skip linting"
            echo "  --no-format  Skip formatting"
            echo "  --release    Build in release mode"
            echo "  --debug      Build in debug mode (default)"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Format code
if [[ $SKIP_FORMAT -eq 0 ]]; then
    echo -e "${GREEN}Formatting code...${NC}"
    if [[ -f "./scripts/format_cpp.sh" ]]; then
        bash ./scripts/format_cpp.sh
    else
        echo -e "${YELLOW}Warning: format_cpp.sh not found. Skipping formatting.${NC}"
    fi
    echo -e "${GREEN}Formatting done.${NC}"
    echo
fi

# Lint code
if [[ $SKIP_LINT -eq 0 ]]; then
    echo -e "${GREEN}Linting code...${NC}"
    if [[ -f "./scripts/lint_cpp.sh" ]]; then
        bash ./scripts/lint_cpp.sh
    else
        echo -e "${YELLOW}Warning: lint_cpp.sh not found. Skipping linting.${NC}"
    fi
    echo -e "${GREEN}Linting done.${NC}"
    echo
fi

# Build the project
if [[ $SKIP_BUILD -eq 0 ]]; then
    echo -e "${GREEN}Building project...${NC}"
    
    BUILD_ARGS=""
    if [[ $CLEAN_BUILD -eq 1 ]]; then
        BUILD_ARGS="--clean"
    fi
    
    if [[ "$BUILD_TYPE" == "Release" ]]; then
        BUILD_ARGS="$BUILD_ARGS --release"
    fi
    
    if [[ -f "./scripts/build.sh" ]]; then
        bash ./scripts/build.sh $BUILD_ARGS
    else
        # Fallback if build script not available
        echo -e "${YELLOW}Warning: build.sh not found. Using direct CMake commands.${NC}"
        
        BUILD_DIR="$PROJECT_ROOT/build"
        if [[ $CLEAN_BUILD -eq 1 && -d "$BUILD_DIR" ]]; then
            echo -e "${YELLOW}Cleaning build directory...${NC}"
            rm -rf "$BUILD_DIR"
        fi
        
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
        make -j$(nproc)
        cd "$PROJECT_ROOT"
    fi
    
    echo -e "${GREEN}Build complete.${NC}"
    echo
fi

# Run tests
if [[ $SKIP_TEST -eq 0 ]]; then
    echo -e "${GREEN}Running tests...${NC}"
    
    if [[ -d "$PROJECT_ROOT/build" ]]; then
        cd "$PROJECT_ROOT/build"
        
        # Run C++ tests if they exist
        if command -v ctest &> /dev/null; then
            echo -e "${GREEN}Running C++ tests with CTest...${NC}"
            ctest --output-on-failure
        else
            echo -e "${YELLOW}Warning: CTest not found. Skipping C++ tests.${NC}"
            
            # Try to find and run test executables directly
            TEST_EXECS=$(find . -type f -executable -name "test_*")
            if [[ -n "$TEST_EXECS" ]]; then
                echo -e "${GREEN}Found test executables, running them directly...${NC}"
                for test_exec in $TEST_EXECS; do
                    echo "Running $test_exec..."
                    $test_exec
                done
            fi
        fi
        
        cd "$PROJECT_ROOT"
    else
        echo -e "${YELLOW}Warning: Build directory not found. Run with --no-test or build first.${NC}"
    fi
    
    # Run Python tests if they exist
    if [[ -d "$PROJECT_ROOT/python/tests" ]]; then
        echo -e "${GREEN}Running Python tests...${NC}"
        if command -v pytest &> /dev/null; then
            python -m pytest "$PROJECT_ROOT/python/tests" -v
        else
            echo -e "${YELLOW}Warning: pytest not found. Skipping Python tests.${NC}"
        fi
    fi
    
    echo -e "${GREEN}All tests completed.${NC}"
fi

echo -e "${GREEN}All done! 🚀${NC}"