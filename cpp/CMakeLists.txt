cmake_minimum_required(VERSION 3.24)
project(srl_sim_cpp VERSION 1.0.0 LANGUAGES C CXX)

# C++ port of the MSc MuJoCo SRL simulation. See docs/03-cpp-design.md.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(SRL_WITH_VIEWER "Build the GLFW viewer" ON)
option(SRL_WITH_OSQP   "Build the OSQP safety-projection fallback" ON)
option(SRL_WITH_GPMP2  "Build the optional GPMP2 joint-space planner" OFF)
option(SRL_BUILD_TESTS "Build unit tests" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(SrlVenvDeps)
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# --------------------------------------------------------------------- Eigen
# Pinocchio's headers require Eigen and this machine has none installed.
# Pinned, fetched into the build tree, never installed system-wide.
FetchContent_Declare(eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
  DOWNLOAD_EXTRACT_TIMESTAMP ON)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()
add_library(srl_eigen INTERFACE)
target_include_directories(srl_eigen SYSTEM INTERFACE "${eigen_SOURCE_DIR}")

# ------------------------------------------------------------ urdfdom_headers
# Pinocchio's parsers/mjcf.hpp includes parsers/urdf.hpp unconditionally, which
# needs <urdf_model/model.h>. The venv ships liburdfdom_model but not the
# headers package, so fetch the header-only repo. We never call the URDF
# parser -- these headers only have to satisfy declarations.
FetchContent_Declare(urdfdom_headers
  GIT_REPOSITORY https://github.com/ros/urdfdom_headers.git
  GIT_TAG 3.0.0
  GIT_SHALLOW ON)
FetchContent_GetProperties(urdfdom_headers)
if(NOT urdfdom_headers_POPULATED)
  FetchContent_Populate(urdfdom_headers)
endif()
# 3.0.0 keeps a single top-level include/; older layouts used <pkg>/include.
file(GLOB _srl_urdf_include_dirs
  "${urdfdom_headers_SOURCE_DIR}/include"
  "${urdfdom_headers_SOURCE_DIR}/*/include")
if(NOT _srl_urdf_include_dirs)
  message(FATAL_ERROR "no include dir in ${urdfdom_headers_SOURCE_DIR}")
endif()
add_library(srl_urdfdom_headers INTERFACE)
target_include_directories(srl_urdfdom_headers SYSTEM INTERFACE
  ${_srl_urdf_include_dirs})

# --------------------------------------------------------------------- toml++
FetchContent_Declare(tomlplusplus
  GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
  GIT_TAG v3.4.0
  GIT_SHALLOW ON)
FetchContent_MakeAvailable(tomlplusplus)

# --------------------------------------------------------------------- LAPACK
# Keep Apple parity by using the same Accelerate LAPACK as NumPy. Ubuntu uses
# its system LAPACK implementation; exact Python bit parity is an Apple-only
# verification target, while the equations and units are portable.
if(APPLE)
  find_library(SRL_LAPACK Accelerate REQUIRED)
else()
  find_package(LAPACK REQUIRED)
  set(SRL_LAPACK LAPACK::LAPACK)
endif()

# --------------------------------------------------------------------- GPMP2
# GPMP2 is intentionally optional. The copy currently available to this
# project is an Ubuntu x86-64 build and cannot be linked by the macOS parity
# build. On Ubuntu, point SRL_GPMP2_ROOT at a prefix containing include/ and
# lib/libgpmp2.so plus lib/libgtsam.so.
if(SRL_WITH_GPMP2)
  set(SRL_GPMP2_ROOT "" CACHE PATH "GPMP2/GTSAM installation prefix")
  find_path(SRL_GPMP2_INCLUDE_DIR gpmp2/gp/GaussianProcessPriorLinear.h
    HINTS "${SRL_GPMP2_ROOT}/include")
  find_library(SRL_GPMP2_LIBRARY NAMES gpmp2
    HINTS "${SRL_GPMP2_ROOT}/lib")
  find_library(SRL_GTSAM_LIBRARY NAMES gtsam
    HINTS "${SRL_GPMP2_ROOT}/lib")
  if(NOT SRL_GPMP2_INCLUDE_DIR OR
     NOT SRL_GPMP2_LIBRARY OR
     NOT SRL_GTSAM_LIBRARY)
    message(FATAL_ERROR
      "SRL_WITH_GPMP2=ON requires GPMP2 and GTSAM. "
      "Pass -DSRL_GPMP2_ROOT=<prefix>.")
  endif()
  get_filename_component(SRL_GPMP2_LIBRARY_DIR "${SRL_GPMP2_LIBRARY}" DIRECTORY)
  list(APPEND SRL_RPATHS "${SRL_GPMP2_LIBRARY_DIR}")
endif()

# ----------------------------------------------------------------------- OSQP
if(SRL_WITH_OSQP)
  FetchContent_Declare(osqp
    GIT_REPOSITORY https://github.com/osqp/osqp.git
    GIT_TAG v1.0.0
    GIT_SHALLOW ON)
  set(OSQP_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
  set(OSQP_BUILD_DEMO_EXE OFF CACHE BOOL "" FORCE)
  set(OSQP_ENABLE_PROFILING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(osqp)
endif()

# ---------------------------------------------------------------------- GLFW
if(SRL_WITH_VIEWER)
  FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW ON)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(glfw)
endif()

# ------------------------------------------------------------------- library
file(GLOB_RECURSE SRL_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
list(FILTER SRL_SOURCES EXCLUDE REGEX "/src/app/")
list(FILTER SRL_SOURCES EXCLUDE REGEX "/src/planning/Gpmp2Planner\\.cpp$")
if(NOT SRL_WITH_VIEWER)
  list(FILTER SRL_SOURCES EXCLUDE REGEX "/src/render/")
endif()

add_library(srl_core ${SRL_SOURCES})
target_include_directories(srl_core PUBLIC "${CMAKE_CURRENT_LIST_DIR}/src")
target_link_libraries(srl_core
  PUBLIC srl_eigen srl_urdfdom_headers srl::mujoco srl::pinocchio
  PRIVATE tomlplusplus::tomlplusplus ${SRL_LAPACK})
target_compile_definitions(srl_core PUBLIC
  SRL_PYTHON_ROOT="${CMAKE_CURRENT_LIST_DIR}/..")
target_compile_options(srl_core PRIVATE -Wall -Wextra -Wpedantic)

if(SRL_WITH_OSQP)
  # Built through add_subdirectory, so the plain target name applies -- the
  # osqp::osqpstatic alias only exists for an installed OSQP.
  if(TARGET osqp::osqpstatic)
    target_link_libraries(srl_core PRIVATE osqp::osqpstatic)
  else()
    target_link_libraries(srl_core PRIVATE osqpstatic)
  endif()
  target_include_directories(srl_core PRIVATE
    "${osqp_SOURCE_DIR}/include/public"
    "${osqp_BINARY_DIR}/include/public"
    "${osqp_BINARY_DIR}/codegen_src/inc/public")
  target_compile_definitions(srl_core PUBLIC SRL_HAVE_OSQP=1)
endif()
if(SRL_WITH_VIEWER)
  target_link_libraries(srl_core PUBLIC glfw)
  target_compile_definitions(srl_core PUBLIC SRL_HAVE_VIEWER=1)
endif()
set_target_properties(srl_core PROPERTIES
  BUILD_RPATH "${SRL_RPATHS}"
  INSTALL_RPATH "${SRL_RPATHS}")

# Keep the older dependency headers bundled with HumanSL's GPMP2 build away
# from the current Pinocchio headers used by srl_core.
if(SRL_WITH_GPMP2)
  add_library(srl_gpmp2 src/planning/Gpmp2Planner.cpp)
  target_include_directories(srl_gpmp2
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/src"
    SYSTEM PRIVATE "${SRL_GPMP2_INCLUDE_DIR}")
  target_link_libraries(srl_gpmp2
    PUBLIC srl_core
    PRIVATE "${SRL_GPMP2_LIBRARY}" "${SRL_GTSAM_LIBRARY}")
  target_compile_definitions(srl_gpmp2 PUBLIC SRL_HAVE_GPMP2=1)
  target_compile_options(srl_gpmp2 PRIVATE -Wall -Wextra -Wpedantic)
  set_target_properties(srl_gpmp2 PROPERTIES
    BUILD_RPATH "${SRL_RPATHS}"
    INSTALL_RPATH "${SRL_RPATHS}")
endif()

# ---------------------------------------------------------------------- apps
function(srl_add_app name source)
  if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/${source}")
    add_executable(${name} "${source}")
    target_link_libraries(${name} PRIVATE srl_core)
    set_target_properties(${name} PROPERTIES
      BUILD_RPATH "${SRL_RPATHS}" INSTALL_RPATH "${SRL_RPATHS}")
  endif()
endfunction()

srl_add_app(srl_smoke        src/app/smoke.cpp)
srl_add_app(srl_print_config src/app/print_config.cpp)
srl_add_app(srl_sim          src/app/main.cpp)
srl_add_app(srl_golden_trace src/app/golden_trace.cpp)
srl_add_app(srl_trajectory_dump src/app/trajectory_dump.cpp)
srl_add_app(srl_headless_trace src/app/headless_trace.cpp)
if(SRL_WITH_GPMP2)
  add_executable(srl_gpmp2_plan src/app/gpmp2_plan.cpp)
  add_executable(srl_gpmp2_sim src/app/gpmp2_sim.cpp)
  foreach(_srl_gpmp2_app srl_gpmp2_plan srl_gpmp2_sim)
    target_link_libraries(${_srl_gpmp2_app} PRIVATE srl_gpmp2)
    set_target_properties(${_srl_gpmp2_app} PROPERTIES
      BUILD_RPATH "${SRL_RPATHS}" INSTALL_RPATH "${SRL_RPATHS}")
  endforeach()
endif()

# --------------------------------------------------------------------- tests
if(SRL_BUILD_TESTS)
  enable_testing()
  file(GLOB SRL_TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/tests/test_*.cpp")
  foreach(_srl_test ${SRL_TEST_SOURCES})
    get_filename_component(_srl_test_name "${_srl_test}" NAME_WE)
    add_executable(${_srl_test_name} "${_srl_test}")
    target_link_libraries(${_srl_test_name} PRIVATE srl_core)
    target_include_directories(${_srl_test_name} PRIVATE
      "${CMAKE_CURRENT_LIST_DIR}/tests")
    set_target_properties(${_srl_test_name} PROPERTIES
      BUILD_RPATH "${SRL_RPATHS}" INSTALL_RPATH "${SRL_RPATHS}")
    add_test(NAME ${_srl_test_name} COMMAND ${_srl_test_name})
  endforeach()
endif()
