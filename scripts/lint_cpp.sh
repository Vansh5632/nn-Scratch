#!/bin/bash
set -eo pipefail

# Colors for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to check if a command exists
check_command() {
  if ! command -v "$1" &> /dev/null; then
    echo -e "${YELLOW}$1 not found. Installing...${NC}"
    return 1
  else
    echo -e "${GREEN}$1 found.${NC}"
    return 0
  fi
}

# Function to check and install dependencies
check_dependencies() {
  local missing_deps=()
  
  # Check for clang-format
  if ! check_command clang-format; then
    missing_deps+=("clang-format")
  fi
  
  # Check for clang-tidy
  if ! check_command clang-tidy; then
    missing_deps+=("clang-tidy")
  fi
  
  # Check for cppcheck
  if ! check_command cppcheck; then
    missing_deps+=("cppcheck")
  fi
  
  # Check for cmake
  if ! check_command cmake; then
    missing_deps+=("cmake")
  fi
  
  # Install missing dependencies if any
  if [ ${#missing_deps[@]} -ne 0 ]; then
    echo -e "${YELLOW}Installing missing dependencies: ${missing_deps[*]}${NC}"
    sudo apt-get update
    sudo apt-get install -y "${missing_deps[@]}"
  fi
}

# Function to format C++ code
format_code() {
  echo -e "${GREEN}Formatting C++ code...${NC}"
  
  if find include src -type f \( -name "*.h" -o -name "*.cpp" \) 2>/dev/null | grep -v "third_party" > /dev/null; then
    find include src -type f \( -name "*.h" -o -name "*.cpp" \) | grep -v "third_party" | xargs clang-format -style=file -i
    echo -e "${GREEN}Code formatting applied.${NC}"
  else
    echo -e "${YELLOW}No .cpp or .h files found in include/ or src/ directories.${NC}"
  fi
}

# Function to run clang-tidy
run_clang_tidy() {
  echo -e "${GREEN}Running clang-tidy...${NC}"
  
  # Configure CMake to generate compile_commands.json
  if [ ! -d "build" ]; then
    mkdir -p build
  fi
  
  cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  
  local tidy_issues=0
  if find include src -type f \( -name "*.h" -o -name "*.cpp" \) 2>/dev/null | grep -v "third_party" > /dev/null; then
    find include src -type f \( -name "*.h" -o -name "*.cpp" \) | grep -v "third_party" | xargs clang-tidy -p build --warnings-as-errors=*
    tidy_issues=$?
  else
    echo -e "${YELLOW}No .cpp or .h files found in include/ or src/ directories.${NC}"
  fi
  
  if [ $tidy_issues -ne 0 ]; then
    echo -e "${RED}Clang-tidy check failed.${NC}"
  else
    echo -e "${GREEN}Clang-tidy check passed.${NC}"
  fi
  
  return $tidy_issues
}

# Function to run cppcheck
run_cppcheck() {
  echo -e "${GREEN}Running cppcheck...${NC}"
  
  local check_issues=0
  if [ -d "include" ] || [ -d "src" ]; then
    cppcheck_dirs=()
    [ -d "include" ] && cppcheck_dirs+=("include/")
    [ -d "src" ] && cppcheck_dirs+=("src/")
    
    cppcheck --enable=all --inconclusive --std=c++14 --suppress=missingInclude --quiet -I include "${cppcheck_dirs[@]}"
    check_issues=$?
  else
    echo -e "${YELLOW}Neither include/ nor src/ directories found.${NC}"
  fi
  
  if [ $check_issues -ne 0 ]; then
    echo -e "${RED}Cppcheck failed.${NC}"
  else
    echo -e "${GREEN}Cppcheck passed.${NC}"
  fi
  
  return $check_issues
}

# Main function
main() {
  echo -e "${GREEN}=====================${NC}"
  echo -e "${GREEN}C++ Linting Script${NC}"
  echo -e "${GREEN}=====================${NC}"
  
  # Check dependencies
  check_dependencies
  
  # Initialize overall exit code
  local exit_code=0
  
  # Format the code (apply changes directly)
  format_code
  
  # Run clang-tidy
  run_clang_tidy
  local tidy_code=$?
  [ $tidy_code -ne 0 ] && exit_code=1
  
  # Run cppcheck
  run_cppcheck
  local check_code=$?
  [ $check_code -ne 0 ] && exit_code=1
  
  if [ $exit_code -ne 0 ]; then
    echo -e "${RED}Linting checks failed. Please fix the issues before committing.${NC}"
  else
    echo -e "${GREEN}All linting checks passed!${NC}"
  fi
  
  return $exit_code
}

# Run the main function
main "$@"
