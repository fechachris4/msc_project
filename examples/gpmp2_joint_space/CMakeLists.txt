cmake_minimum_required(VERSION 3.16)
project(gpmp2_joint_space_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GPMP2_ROOT "/usr/local" CACHE PATH "GPMP2/GTSAM installation prefix")

find_path(
    GPMP2_INCLUDE_DIR
    NAMES gpmp2/kinematics/ArmModel.h
    HINTS "${GPMP2_ROOT}/include"
    REQUIRED
)
find_library(
    GPMP2_LIBRARY
    NAMES gpmp2
    HINTS "${GPMP2_ROOT}/lib"
    REQUIRED
)
find_path(
    GTSAM_INCLUDE_DIR
    NAMES gtsam/nonlinear/NonlinearFactorGraph.h
    HINTS "${GPMP2_ROOT}/include"
    REQUIRED
)
find_library(
    GTSAM_LIBRARY
    NAMES gtsam
    HINTS "${GPMP2_ROOT}/lib"
    REQUIRED
)
find_path(
    EIGEN3_INCLUDE_DIR
    NAMES Eigen/Core
    HINTS
        "${GPMP2_ROOT}/include/eigen3"
        "/usr/include/eigen3"
    REQUIRED
)
find_path(
    BOOST_INCLUDE_DIR
    NAMES boost/concept_check.hpp
    HINTS
        "${GPMP2_ROOT}/include"
        "/opt/homebrew/include"
        "/usr/include"
    REQUIRED
)

add_executable(gpmp2_joint_space_demo main.cpp)
target_include_directories(
    gpmp2_joint_space_demo
    PRIVATE
        "${GPMP2_INCLUDE_DIR}"
        "${GTSAM_INCLUDE_DIR}"
        "${EIGEN3_INCLUDE_DIR}"
        "${BOOST_INCLUDE_DIR}"
)
target_link_libraries(
    gpmp2_joint_space_demo
    PRIVATE
        "${GPMP2_LIBRARY}"
        "${GTSAM_LIBRARY}"
)
