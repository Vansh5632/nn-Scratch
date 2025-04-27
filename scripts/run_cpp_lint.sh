#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Ensure we have a compile_commands.json
if [ ! -f "${PROJECT_ROOT}/build/compile_commands.json" ]; then
  echo "Creating compile_commands.json..."
  cmake -S "${PROJECT_ROOT}" -B "${PROJECT_ROOT}/build" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
fi

echo "=== Step 1: Checking code formatting ==="
find "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/src" -name "*.h" -o -name "*.cpp" | grep -v "third_party" | xargs clang-format -style=file --dry-run --Werror || {
  echo "Formatting issues found. Running formatter..."
  "${SCRIPT_DIR}/format_cpp.sh"
  echo "Code formatting fixed."
}

echo "=== Step 2: Running cppcheck static analysis ==="
cppcheck --enable=all --inconclusive --std=c++14 --suppress=missingInclude --quiet -I "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/include/" "${PROJECT_ROOT}/src/"

echo "=== Step 3: Running clang-tidy analysis ==="
find "${PROJECT_ROOT}/include" "${PROJECT_ROOT}/src" -name "*.h" -o -name "*.cpp" | grep -v "third_party" | xargs clang-tidy -p "${PROJECT_ROOT}/build"

echo "All linting checks completed successfully!"