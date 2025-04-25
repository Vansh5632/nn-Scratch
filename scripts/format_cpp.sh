#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null; then
  echo "Error: clang-format not found. Please install it."
  exit 1
fi

# Format files
echo "Formatting C++ files with clang-format..."

# Find all C++ files excluding third-party code
find "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/src" -type f \( -name "*.h" -o -name "*.cpp" \) | \
  grep -v "third_party" | \
  xargs clang-format -i -style=file

echo "Formatting complete!"