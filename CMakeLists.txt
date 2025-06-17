cmake_minimum_required(VERSION 3.17)

# Project declaration
project(TorchScratch
    VERSION 0.1.0
    DESCRIPTION "A custom neural network library built from scratch"
    LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable Position Independent Code for all targets
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enable compilation database for tools like clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Platform detection
if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include FetchContent module for dependencies
include(FetchContent)

# Download and configure pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Find Python - needed for pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Get the Python site-packages directory for installation
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Include static analyzer options
include(cmake/StaticAnalyzers.cmake)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${pybind11_SOURCE_DIR}/include  # Add pybind11 include path
)

# Create library target
add_library(torchscratch 
    src/core/tensor/tensor.cpp 
    src/core/tensor/ops.cpp 
    src/core/tensor/tensor_impl.cpp
    src/core/autograd/engine.cpp
    src/core/nn/linear.cpp
    src/core/nn/activation.cpp
    src/core/nn/loss.cpp
    src/core/optim/sgd.cpp
)

# Explicitly set Position Independent Code for the library
set_property(TARGET torchscratch PROPERTY POSITION_INDEPENDENT_CODE ON)

set(HEADER
    include/core/tensor/tensor.h
    include/core/tensor/ops.h
    include/core/tensor/tensor_impl.h
    include/core/autograd/function.h
    include/core/autograd/variable.h)

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

# Python bindings module
pybind11_add_module(torchscratch_cpp
    src/python/module.cpp
    src/python/tensor.cpp
    src/python/nn.cpp
    src/python/optim.cpp
)
target_link_libraries(torchscratch_cpp PRIVATE torchscratch)

# Enable testing
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    
    # Download and configure Google Test
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
        test/cpp/autograd/test_autograd.cpp
    )

    # Create separate test executables for tensor and autograd
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/cpp/tensor/test_tensor.cpp)
        add_executable(test_tensor test/cpp/tensor/test_tensor.cpp)
        target_link_libraries(test_tensor torchscratch gtest gtest_main)
        add_test(NAME TensorTests COMMAND test_tensor)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/cpp/autograd/test_autograd.cpp)
        add_executable(test_autograd test/cpp/autograd/test_autograd.cpp)
        target_link_libraries(test_autograd torchscratch gtest gtest_main)
        add_test(NAME AutogradTests COMMAND test_autograd)
    endif()
endif()

# Add format target
add_custom_target(
    format
    COMMAND ${CMAKE_COMMAND} -E echo "Formatting code..."
    COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/format_cpp.sh || (echo "Formatting failed but continuing" && /bin/true)
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Formatting C++ code with clang-format"
    VERBATIM
)

# Add lint target
add_custom_target(
    lint
    COMMAND ${CMAKE_COMMAND} -E echo "Linting code..."
    COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/lint_cpp.sh || (echo "Linting found issues but continuing" && /bin/true)
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Linting C++ code with clang-tidy and cppcheck"
    VERBATIM
)

# Installation rules
include(GNUInstallDirs)

# Install library and headers
install(TARGETS torchscratch
    EXPORT torchscratch-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ 
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS torchscratch_cpp
    LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}
    ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
)

# Export targets
install(EXPORT torchscratch-targets
    FILE torchscratch-targets.cmake
    NAMESPACE torchscratch::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/torchscratch
)

# Create and install package config
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/torchscratch-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/torchscratch-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/torchscratch
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/torchscratch-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/torchscratch-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/torchscratch-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/torchscratch
)