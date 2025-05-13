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
add_library(torchscratch src/core/tensor/tensor.cpp src/core/tensor/ops.cpp)

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

# Clang-Tidy configuration
option(ENABLE_CLANG_TIDY "Enable clang-tidy static analysis" OFF)
if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy)
    if(CLANG_TIDY_EXE)
    message(STATUS "Found clang-tidy: ${CLANG_TIDY_EXE}")
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
    else()
    message(WARNING "clang-tidy requested but not found")
    endif()
endif()

# Cppcheck configuration
option(ENABLE_CPPCHECK "Enable cppcheck static analysis" OFF)
if(ENABLE_CPPCHECK)
    find_program(CPPCHECK_EXE NAMES cppcheck)
    if(CPPCHECK_EXE)
    message(STATUS "Found cppcheck: ${CPPCHECK_EXE}")
    set(CMAKE_CXX_CPPCHECK 
        "${CPPCHECK_EXE}" 
        "--enable=warning,performance,portability" 
        "--std=c++14"
        "--suppress=missingIncludeSystem"
        "--quiet"
        "--inline-suppr"
    )
    else()
    message(WARNING "cppcheck requested but not found")
    endif()
endif()

# Add common compiler warnings
function(set_project_warnings project_name)
    set(MSVC_WARNINGS
    /W4     # Baseline reasonable warnings
    /permissive-    # standards conformance
    /w14242 # 'identifier': conversion, possible loss of data
    /w14254 # 'operator': conversion, possible loss of data
    /w14263 # function does not override any base class virtual member function
    /w14265 # class has virtual functions, but destructor is not virtual
    /w14287 # unsigned/negative constant mismatch
    /w14289 # loop control variable is used outside the for-loop scope
    /w14296 # expression is always 'boolean_value'
    /w14311 # pointer truncation from 'type1' to 'type2'
    /w14545 # expression before comma evaluates to a function which is missing an argument list
    /w14546 # function call before comma missing argument list
    /w14547 # 'operator': operator before comma has no effect
    /w14549 # 'operator': operator before comma has no effect; expected operator with side-effect
    /w14555 # expression has no effect
    /w14619 # pragma warning: there is no warning number 'number'
    /w14640 # 'instance': construction of local static object is not thread-safe
    /w14826 # Conversion from 'type1' to 'type2' is sign-extended
    /w14905 # wide string literal cast to 'LPSTR'
    /w14906 # string literal cast to 'LPWSTR'
    )

    set(GCC_CLANG_WARNINGS
    -Wall
    -Wextra    # reasonable and standard
    -Wpedantic # warn if non-standard C++ is used
    -Wshadow   # warn the user if a variable declaration shadows one from a parent context
    -Wcast-align    # warn for potential performance problem casts
    -Wunused        # warn on anything being unused
    -Wconversion    # warn on type conversions that may lose data
    -Wsign-conversion   # warn on sign conversions
    -Wnull-dereference  # warn if a null dereference is detected
    -Wdouble-promotion  # warn if float is implicitly promoted to double
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    target_compile_options(${project_name} INTERFACE ${GCC_CLANG_WARNINGS})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${project_name} INTERFACE ${GCC_CLANG_WARNINGS})
    elseif(MSVC)
    target_compile_options(${project_name} INTERFACE ${MSVC_WARNINGS})
    endif()
endfunction()

# Apply warnings to the torchscratch target
set_project_warnings(torchscratch)

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