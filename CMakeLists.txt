cmake_minimum_required(VERSION 3.14)

# Project declaration
project(TorchScratch
        VERSION 0.1.0
        DESCRIPTION "A custom neural network library"
        LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable compilation database for tools like clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include static analyzer options
include(cmake/StaticAnalyzers.cmake)

# Define header files (for IDE indexing)
set(HEADERS
    include/core/tensor/tensor.h
    # Add other header files here
)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# Create library target
add_library(torchscratch src/core/tensor/tensor.cpp)

# Set include directories for the target
target_include_directories(torchscratch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Set compile options for the target (warnings, etc.)
target_compile_options(torchscratch PRIVATE
    -Wall
    -Wextra
    # -Werror
    -Wno-unused-parameter
)

# Add format target
add_custom_target(
    format
    COMMAND ${CMAKE_COMMAND} -E echo "Formatting code..."
    COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/format_cpp.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Formatting C++ code with clang-format"
    VERBATIM
)

# Add lint target
add_custom_target(
    lint
    COMMAND ${CMAKE_COMMAND} -E echo "Linting code..."
    COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/lint_cpp.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Linting C++ code with clang-tidy and cppcheck"
    VERBATIM
)

# Download and configure Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Test executable
set(TEST_SOURCES
    test/cpp/tensor/test_tensor.cpp
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/cpp/tensor/test_tensor.cpp)
  add_executable(test_tensor ${TEST_SOURCES})
  target_link_libraries(test_tensor torchscratch gtest gtest_main)

  # Enable testing
  enable_testing()
  add_test(NAME TensorTests COMMAND test_tensor)
endif()