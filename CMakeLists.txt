CMAKE_MINIMUM_REQUIRED(VERSION 3.20 FATAL_ERROR)
MESSAGE(STATUS "Using CMake ${CMAKE_VERSION}")

SET(CUBEAILIB_VERSION_MAJOR 0)
SET(CUBEAILIB_VERSION_MINOR 4)
SET(CUBEAILIB_VERSION_PATCH 3)
SET(CUBEAILIB_VERSION "${CUBEAILIB_VERSION_MAJOR}.${CUBEAILIB_VERSION_MINOR}.${CUBEAILIB_VERSION_PATCH}")


MESSAGE(STATUS "cuberl lib Version ${CUBEAILIB_VERSION}")


# if PyTorch has been compiled with CUDA support
# then we need to enable this flag
SET(USE_CUDA OFF)

IF(USE_CUDA)
	PROJECT(cubeai VERSION ${CUBEAILIB_VERSION} LANGUAGES CXX C CUDA)
	MESSAGE(STATUS "Compiling with CUDA support")
ELSE()
	PROJECT(cubeai VERSION ${CUBEAILIB_VERSION} LANGUAGES CXX C)
	MESSAGE(STATUS "Compiling without CUDA support")
ENDIF()


# -----------------------------------------------------------------------------
# Prevent in-source builds.
# -----------------------------------------------------------------------------

IF(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	MESSAGE( FATAL_ERROR  "In-source build is not possible. Choose an empty directory for build output.")
ENDIF(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})

# Be sure to avoid troubles with library paths when using old policy
IF(COMMAND cmake_policy)
	CMAKE_POLICY(SET CMP0003 NEW)
	CMAKE_POLICY(SET CMP0048 NEW)
ENDIF(COMMAND cmake_policy)

OPTION(ENABLE_WEBOTS_FLAG OFF)

# configure the project accordingly
# Set the build type options supported
# Debug and Release are the types supported
SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_COMPILER g++-11)
SET(CMAKE_CXX_STANDARD 20)
#SET(CMAKE_CXX_STANDARD_REQUIRED True)
#SET(CMAKE_C_COMPILER gcc-11)
#SET(CMAKE_C_STANDARD 20)
#SET(CMAKE_C_STANDARD_REQUIRED True)
SET(CMAKE_LINKER_FLAGS "-pthread")

SET(USE_OPENMP ON)
SET(USE_LOG ON)
SET(USE_PYTORCH ON)
SET(USE_OPENCV OFF)
SET(RLENVS_CPP_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/external/rlenvs_cpp/src") # required

SET(USE_RL ON)
SET(DATA_SET_FOLDER ${PROJECT_SOURCE_DIR}/data)
SET(ENABLE_EXAMPLES_FLAG ON)
SET(ENABLE_TESTS_FLAG ON) # Testing
SET(ENABLE_DOC_FLAG OFF) # Documentation

# if we enable Webots then specify the paths
# Note that if rlenvscpp is compiled with
# Webots then this flags has to be ON
IF(ENABLE_WEBOTS_FLAG)
	# we have webots include the directories
    SET(WEBOTS_LIB_DIRS "/home/alex/qi3/rlenvs_from_cpp/external/webots/lib/controller")    
    SET(WEBOTS_INCLUDE_DIRS "/home/alex/qi3/rlenvs_from_cpp/external/webots/include/controller/cpp")
	
	INCLUDE_DIRECTORIES(${WEBOTS_INCLUDE_DIRS})
	LINK_DIRECTORIES(${WEBOTS_LIB_DIRS})
	SET(USE_CUBERL_WEBOTS ON)
ELSE()
	MESSAGE( WARNING  "Building without Webots.")
ENDIF()


IF(USE_CUDA)
	# The CUDA compiler. You may need to set this manually
	SET(CMAKE_CUDA_COMPILER "/usr/bin/nvcc")
	
	# we need to be careful here as PyTorch may not
	# accept what we set here. PyTorch version ?
	# needs at least C++17 to work
	SET(CMAKE_CUDA_STANDARD 17)
	SET(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS}; "-O3 ${ARCH_FLAGS}")
	
ENDIF()

# if we use PyTorch set up the path for libtorch
IF(USE_PYTORCH)
	#SET(PYTORCH_PATH "/home/alex/MySoftware/libtorch")
	#SET(PYTORCH_PATH "/home/alex/MySoftware/libtorch_cu118")
	SET(PYTORCH_PATH "/home/alex/MySoftware/libtorch_cpu/libtorch")
ENDIF()

IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
	SET(CMAKE_INSTALL_PREFIX  ${PROJECT_SOURCE_DIR}/install/dbg)
	SET(CMAKE_CXX_FLAGS "-g ") # -Wall -Wextra")
	SET(CUBERL_DEBUG ON)
	SET(CUBERL_PRINT_DBG_MSGS ON)
	SET(BOOST_LOG_DYN_LINK ON)
ELSEIF(CMAKE_BUILD_TYPE STREQUAL "Release")
	SET(CMAKE_INSTALL_PREFIX  ${PROJECT_SOURCE_DIR}/install/opt)
	SET(CMAKE_CXX_FLAGS "-O2")
    SET(CUBEAI_DEBUG OFF)
    SET(BOOST_LOG_DYN_LINK ON)
ENDIF()


IF(CMAKE_BUILD_TYPE MATCHES "Debug")
	SET(RLENVS_CPP_LIB_PATH "${PROJECT_SOURCE_DIR}/external/rlenvs_cpp/install/dbg")
ELSE()
	SET(RLENVS_CPP_LIB_PATH "${PROJECT_SOURCE_DIR}/external/rlenvs_cpp/install/opt")
ENDIF()


# if tests are enabled try to
# find GTest
IF(ENABLE_TESTS_FLAG)
	# find gtest
	FIND_PACKAGE(GTest REQUIRED)

	IF(GTest_FOUND)
		INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS})
		LINK_DIRECTORIES("/usr/local/lib")
    ELSE()
	    MESSAGE( FATAL_ERROR  "ENABLE_TESTS_FLAG is ON but could not find GTest library.")
	ENDIF()
ENDIF()

# find needed packages
# find Boost, BLAS
FIND_PACKAGE(Boost 1.74.0 REQUIRED)
FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)

IF(Boost_FOUND)
	MESSAGE( STATUS  "Found needed Boost C++ library.")
	SET(Boost_USE_SHARED_LIBS ON)
	SET(Boost_USE_STATIC_LIBS ON)
ELSE()
	MESSAGE( FATAL_ERROR  "Boost C++ library is required but not found.")
ENDIF()

IF(NOT BLAS_FOUND)
	MESSAGE( FATAL_ERROR  "Could not find OpenBLAS library.")
ELSE()
	MESSAGE( STATUS  "Found needed BLAS library.")
ENDIF()

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# if using PyTorch append the path to libtorch
IF(USE_PYTORCH)
	LIST(APPEND CMAKE_PREFIX_PATH ${PYTORCH_PATH})
	FIND_PACKAGE(Torch REQUIRED CONFIG)
	MESSAGE(STATUS "TORCH Include directory ${TORCH_INCLUDE_DIRS}")
	MESSAGE(STATUS "TORCH Lib directory ${TORCH_LIBRARIES}")
ENDIF()

MESSAGE(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
MESSAGE(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
MESSAGE(STATUS "Project SRC directory: ${PROJECT_SOURCE_DIR}")

# write configuration and version files
configure_file(config.h.in ${PROJECT_SOURCE_DIR}/include/cubeai/base/cubeai_config.h @ONLY)
configure_file(version.h.in ${PROJECT_SOURCE_DIR}/include/cubeai/base/cubeai_version.h @ONLY)

# include directories
INCLUDE_DIRECTORIES(${BOOST_INCLUDEDIR})
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${RLENVS_CPP_INCLUDE_DIRS})

IF(USE_PYTORCH)
	INCLUDE_DIRECTORIES(${TORCH_INCLUDE_DIRS})
ENDIF()

INCLUDE_DIRECTORIES(include/)

# collect the source files for the library
FILE(GLOB SRCS src/cubeai/base/*.cpp
               src/cubeai/rl/*.cpp
               src/cubeai/rl/*/*.cpp
			   src/cubeai/rl/algorithms/*.cpp
			   src/cubeai/rl/algorithms/*/*.cpp
               src/cubeai/utils/*.cpp
               src/cubeai/maths/*.cpp
               src/cubeai/maths/statistics/*.cpp
               src/cubeai/maths/statistics/distributions/*.cpp
               src/cubeai/maths/optimization/*.cpp
               #src/cubeai/planning/*.cpp
	)



ADD_LIBRARY(cubeailib SHARED ${SRCS})
SET_TARGET_PROPERTIES(cubeailib PROPERTIES LINKER_LANGUAGE CXX)
INSTALL(TARGETS cubeailib DESTINATION ${CMAKE_INSTALL_PREFIX})

IF(ENABLE_EXAMPLES_FLAG)
	# Add the examples
	ADD_SUBDIRECTORY(examples)
ELSE()
	MESSAGE(WARNING "Examples have not been enabled")
ENDIF()

IF(ENABLE_TESTS_FLAG)
	# Add the tests
	ADD_SUBDIRECTORY(tests)
ELSE()
	MESSAGE(WARNING "Tests have not been enabled")
ENDIF()

IF(ENABLE_DOC_FLAG)
	# Add the documentation
	ADD_SUBDIRECTORY(doc)
ELSE()
	MESSAGE(WARNING "Documentation will not be generated")
ENDIF()

MESSAGE(STATUS "Installation destination at: ${CMAKE_INSTALL_PREFIX}")
MESSAGE(STATUS "Type make install to build cubeAI")
