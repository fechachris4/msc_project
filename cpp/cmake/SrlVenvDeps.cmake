# Discover the native MuJoCo and Pinocchio that the project's Python venv
# already ships. Linking those exact binaries is what makes numerical parity
# with the Python implementation achievable rather than approximate.
#
# Override with -DSRL_VENV=/path/to/.venv or point the individual _ROOT
# variables at a system install.

include_guard(GLOBAL)

set(SRL_VENV "${CMAKE_CURRENT_LIST_DIR}/../../.venv"
    CACHE PATH "Python venv that ships native MuJoCo and Pinocchio")

# ---------------------------------------------------------------- site-packages
if(NOT SRL_SITE_PACKAGES)
  file(GLOB _srl_site_candidates "${SRL_VENV}/lib/python*/site-packages")
  list(LENGTH _srl_site_candidates _srl_site_count)
  if(_srl_site_count EQUAL 0)
    message(FATAL_ERROR
      "No site-packages under ${SRL_VENV}. Pass -DSRL_VENV=<venv> or "
      "-DSRL_SITE_PACKAGES=<dir>.")
  endif()
  list(GET _srl_site_candidates 0 SRL_SITE_PACKAGES)
endif()
message(STATUS "SRL site-packages: ${SRL_SITE_PACKAGES}")

# ---------------------------------------------------------------------- MuJoCo
set(SRL_MUJOCO_ROOT "${SRL_SITE_PACKAGES}/mujoco" CACHE PATH "MuJoCo root")

find_path(SRL_MUJOCO_INCLUDE_DIR
  NAMES mujoco/mujoco.h
  HINTS "${SRL_MUJOCO_ROOT}/include"
  NO_DEFAULT_PATH)

# The wheel ships a versioned dylib (libmujoco.3.10.0.dylib) and no symlink.
file(GLOB _srl_mujoco_libs
  "${SRL_MUJOCO_ROOT}/libmujoco*.dylib"
  "${SRL_MUJOCO_ROOT}/libmujoco*.so*")
if(_srl_mujoco_libs)
  list(GET _srl_mujoco_libs 0 SRL_MUJOCO_LIBRARY)
endif()

if(NOT SRL_MUJOCO_INCLUDE_DIR OR NOT SRL_MUJOCO_LIBRARY)
  message(FATAL_ERROR
    "MuJoCo not found under ${SRL_MUJOCO_ROOT} "
    "(include=${SRL_MUJOCO_INCLUDE_DIR} lib=${SRL_MUJOCO_LIBRARY})")
endif()

add_library(srl::mujoco UNKNOWN IMPORTED GLOBAL)
set_target_properties(srl::mujoco PROPERTIES
  IMPORTED_LOCATION "${SRL_MUJOCO_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${SRL_MUJOCO_INCLUDE_DIR}")
message(STATUS "SRL MuJoCo: ${SRL_MUJOCO_LIBRARY}")

# ------------------------------------------------------------------- Pinocchio
# Deliberately linked by hand rather than through pinocchioConfig.cmake: that
# config chains find_dependency(Eigen3/urdfdom/Boost/coal), and the venv has no
# Eigen3 config to satisfy it. The dylibs carry their own install names, so the
# linker resolves the rest through the added rpath.
set(SRL_CMEEL_PREFIX "${SRL_SITE_PACKAGES}/cmeel.prefix"
    CACHE PATH "cmeel prefix shipping Pinocchio")

find_path(SRL_PINOCCHIO_INCLUDE_DIR
  NAMES pinocchio/parsers/mjcf.hpp
  HINTS "${SRL_CMEEL_PREFIX}/include"
  NO_DEFAULT_PATH)

foreach(_srl_pin_lib pinocchio_default pinocchio_parsers)
  find_library(SRL_LIB_${_srl_pin_lib}
    NAMES ${_srl_pin_lib}
    HINTS "${SRL_CMEEL_PREFIX}/lib"
    NO_DEFAULT_PATH)
  if(NOT SRL_LIB_${_srl_pin_lib})
    message(FATAL_ERROR "lib${_srl_pin_lib} not found in ${SRL_CMEEL_PREFIX}/lib")
  endif()
  list(APPEND SRL_PINOCCHIO_LIBRARIES "${SRL_LIB_${_srl_pin_lib}}")
endforeach()

if(NOT SRL_PINOCCHIO_INCLUDE_DIR)
  message(FATAL_ERROR "Pinocchio headers not found in ${SRL_CMEEL_PREFIX}/include")
endif()

# Copied verbatim from the shipped pinocchioTargets.cmake
# (INTERFACE_COMPILE_DEFINITIONS). Without the Boost.MPL limits the joint
# variant exceeds boost::variant's default 20-type cap and the headers fail to
# compile. PINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION makes us use the *same*
# explicit instantiations the dylib exports, which is what we want for parity.
# PINOCCHIO_WITH_COLLISION is deliberately omitted: we link default+parsers
# only and never touch coal.
set(SRL_PINOCCHIO_DEFINITIONS
  BOOST_MPL_LIMIT_LIST_SIZE=30
  BOOST_MPL_LIMIT_VECTOR_SIZE=30
  BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
  BOOST_FUSION_INVOKE_MAX_ARITY=12
  PINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION
  PINOCCHIO_WITH_URDFDOM)

add_library(srl::pinocchio INTERFACE IMPORTED GLOBAL)
set_target_properties(srl::pinocchio PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${SRL_PINOCCHIO_INCLUDE_DIR}"
  INTERFACE_COMPILE_DEFINITIONS "${SRL_PINOCCHIO_DEFINITIONS}"
  INTERFACE_LINK_LIBRARIES "${SRL_PINOCCHIO_LIBRARIES}"
  INTERFACE_LINK_DIRECTORIES "${SRL_CMEEL_PREFIX}/lib")
message(STATUS "SRL Pinocchio: ${SRL_PINOCCHIO_LIBRARIES}")

# ------------------------------------------------------------ MuJoCo rpath
# The wheel ships a flat libmujoco.<ver>.dylib whose LC_ID_DYLIB still claims
# "@rpath/mujoco.framework/Versions/A/libmujoco.<ver>.dylib". Python gets away
# with it because its extensions link the flat name via @loader_path, but
# anything linking the dylib directly inherits the framework-shaped install
# name. Rather than install_name_tool every target, materialise that layout
# once as symlinks inside the build tree and put it on the rpath.
get_filename_component(_srl_mujoco_libname "${SRL_MUJOCO_LIBRARY}" NAME)
set(SRL_MUJOCO_SHIM_DIR "${CMAKE_BINARY_DIR}/mujoco_rpath")
set(_srl_shim_versions "${SRL_MUJOCO_SHIM_DIR}/mujoco.framework/Versions/A")
file(MAKE_DIRECTORY "${_srl_shim_versions}")
if(NOT EXISTS "${_srl_shim_versions}/${_srl_mujoco_libname}")
  file(CREATE_LINK "${SRL_MUJOCO_LIBRARY}"
       "${_srl_shim_versions}/${_srl_mujoco_libname}" SYMBOLIC)
endif()

# Runtime lookup for all three sets of dylibs.
set(SRL_RPATHS
  "${SRL_CMEEL_PREFIX}/lib"
  "${SRL_MUJOCO_ROOT}"
  "${SRL_MUJOCO_SHIM_DIR}")
