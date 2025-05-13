#!/bin/bash
# filepath: /home/vansh5632/sudo/nn/scripts/format_cpp.sh

set -e

# ANSI color codes
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
NC="\033[0m"  # No Color

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null; then
    echo -e "${YELLOW}Warning: clang-format not found. Please install it to format C++ code.${NC}"
    exit 1
fi

# Define paths to format
INCLUDE_DIR="$PROJECT_ROOT/include"
SRC_DIR="$PROJECT_ROOT/src"
TEST_DIR="$PROJECT_ROOT/test"

echo -e "${GREEN}Formatting C++ code with clang-format...${NC}"

# Find and format all C++ files
find "$INCLUDE_DIR" "$SRC_DIR" "$TEST_DIR" \
    -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.cc" -o -name "*.cxx" \) \
    -exec clang-format -i {} \;

echo -e "${GREEN}Formatting complete!${NC}"