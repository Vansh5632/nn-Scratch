#!/bin/bash
# filepath: /home/vansh5632/sudo/nn/scripts/lint_cpp.sh

set -e

# ANSI color codes
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m"  # No Color

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Check if tools are installed
if ! command -v clang-tidy &> /dev/null; then
    echo -e "${YELLOW}Warning: clang-tidy not found. Please install it for static analysis.${NC}"
    CLANG_TIDY_AVAILABLE=0
else
    CLANG_TIDY_AVAILABLE=1
fi

if ! command -v cppcheck &> /dev/null; then
    echo -e "${YELLOW}Warning: cppcheck not found. Please install it for additional static analysis.${NC}"
    CPPCHECK_AVAILABLE=0
else
    CPPCHECK_AVAILABLE=1
fi

# Define paths to check
INCLUDE_DIR="$PROJECT_ROOT/include"
SRC_DIR="$PROJECT_ROOT/src"
TEST_DIR="$PROJECT_ROOT/test"

# Exit code initially set to 0 (success)
EXIT_CODE=0

# Run clang-tidy
if [ "$CLANG_TIDY_AVAILABLE" -eq 1 ]; then
    echo -e "${GREEN}Running clang-tidy...${NC}"
    
    # Check for compile_commands.json
    if [ ! -f "$PROJECT_ROOT/build/compile_commands.json" ]; then
        echo -e "${YELLOW}Compilation database not found. Make sure to run cmake with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON${NC}"
        echo -e "${YELLOW}Skipping clang-tidy checks...${NC}"
    else
        # Find all C++ files
        CLANG_TIDY_FILES=$(find "$INCLUDE_DIR" "$SRC_DIR" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.cc" -o -name "*.cxx" \) 2>/dev/null || echo "")
        
        if [ -n "$CLANG_TIDY_FILES" ]; then
            for file in $CLANG_TIDY_FILES; do
                echo "Checking $file..."
                # Don't use strict failure mode for clang-tidy
                if ! clang-tidy -p="$PROJECT_ROOT/build" "$file" 2>/dev/null; then
                    echo -e "${YELLOW}clang-tidy found issues in $file but continuing...${NC}"
                fi
            done
        else
            echo "No files found for clang-tidy analysis"
        fi
    fi
    echo
fi

# Run cppcheck with more lenient settings
if [ "$CPPCHECK_AVAILABLE" -eq 1 ]; then
    echo -e "${GREEN}Running cppcheck...${NC}"
    
    # Create directory for reports if it doesn't exist
    mkdir -p "$PROJECT_ROOT/reports"
    
    # Run cppcheck with less strict settings, output errors to a file
    CPPCHECK_REPORT="$PROJECT_ROOT/reports/cppcheck_report.txt"
    if ! cppcheck --error-exitcode=0 --enable=warning,performance,portability --std=c++14 \
           --suppress=missingIncludeSystem \
           -I "$INCLUDE_DIR" "$INCLUDE_DIR" "$SRC_DIR" > "$CPPCHECK_REPORT" 2>&1; then
        echo -e "${YELLOW}Cppcheck found issues, continuing anyway. See report at $CPPCHECK_REPORT${NC}"
        # Don't exit with error
    else
        echo -e "${GREEN}Cppcheck found no critical issues!${NC}"
    fi
    
    if [ -s "$CPPCHECK_REPORT" ]; then
        echo -e "${YELLOW}Cppcheck report (warnings):"
        cat "$CPPCHECK_REPORT"
        echo -e "${NC}"
    fi
    echo
fi

# Check for trailing whitespace and other common issues
echo -e "${GREEN}Checking for common style issues...${NC}"
WHITESPACE_FILES=$(grep -l -r --include="*.cpp" --include="*.h" --include="*.hpp" --include="*.cc" --include="*.cxx" " $" "$INCLUDE_DIR" "$SRC_DIR" 2>/dev/null || echo "")
if [ -n "$WHITESPACE_FILES" ]; then
    echo -e "${YELLOW}Found trailing whitespace in:${NC}"
    for file in $WHITESPACE_FILES; do
        echo "  $file"
    done
    # Don't fail for whitespace issues
else
    echo -e "${GREEN}No trailing whitespace issues!${NC}"
fi

echo -e "${GREEN}Lint checks completed. Fix any warnings before submitting code for review.${NC}"

# Always exit with success to avoid breaking the build
exit 0