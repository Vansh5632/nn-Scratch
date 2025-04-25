# Options for static analysis tools
option(ENABLE_CPPCHECK "Enable static analysis with cppcheck" OFF)
option(ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" OFF)
option(ENABLE_INCLUDE_WHAT_YOU_USE "Enable static analysis with include-what-you-use" OFF)

# Find cppcheck
if(ENABLE_CPPCHECK)
  find_program(CPPCHECK cppcheck)
  if(CPPCHECK)
    set(CMAKE_CXX_CPPCHECK 
        ${CPPCHECK} 
        --suppress=missingInclude 
        --enable=all
        --inline-suppr
        --inconclusive)
    
    # Add other checks
    list(
      APPEND CMAKE_CXX_CPPCHECK
      "--enable=warning"
      "--enable=style"
      "--enable=performance"
      "--enable=portability"
      "--std=c++14"
    )
    
    message(STATUS "Using cppcheck: ${CPPCHECK}")
  else()
    message(FATAL_ERROR "cppcheck requested but executable not found")
  endif()
endif()

# Find clang-tidy
if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY clang-tidy)
  if(CLANG_TIDY)
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY})
    message(STATUS "Using clang-tidy: ${CLANG_TIDY}")
  else()
    message(FATAL_ERROR "clang-tidy requested but executable not found")
  endif()
endif()

# Find include-what-you-use
if(ENABLE_INCLUDE_WHAT_YOU_USE)
  find_program(INCLUDE_WHAT_YOU_USE include-what-you-use)
  if(INCLUDE_WHAT_YOU_USE)
    set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${INCLUDE_WHAT_YOU_USE})
    message(STATUS "Using include-what-you-use: ${INCLUDE_WHAT_YOU_USE}")
  else()
    message(FATAL_ERROR "include-what-you-use requested but executable not found")
  endif()
endif()