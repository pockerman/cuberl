CMAKE_MINIMUM_REQUIRED(VERSION 3.6 FATAL_ERROR)
MESSAGE(STATUS "Using CMake ${CMAKE_VERSION}")


SET(CUBEAILIB_VERSION_MAJOR 0)
SET(CUBEAILIB_VERSION_MINOR 0)
SET(CUBEAILIB_VERSION_PATCH 7)
SET(CUBEAILIB_VERSION "${CUBEAILIB_VERSION_MAJOR}.${CUBEAILIB_VERSION_MINOR}.${CUBEAILIB_VERSION_PATCH}")
PROJECT(cubeai VERSION ${CUBEAILIB_VERSION} LANGUAGES CXX C)
MESSAGE(STATUS "CubeAI lib Version ${CUBEAILIB_VERSION}")

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

# configure the project accordingly

# Set the build type options supported
# Debug and Release
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_COMPILER g++-11)
SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_STANDARD_REQUIRED True)
SET(CMAKE_C_COMPILER gcc-11)
SET(CMAKE_C_STANDARD 20)
SET(CMAKE_C_STANDARD_REQUIRED True)
SET(CMAKE_LINKER_FLAGS "-pthread")

SET(USE_OPENMP ON)
SET(USE_LOG ON)
SET(USE_PYTORCH OFF)
SET(USE_OPENCV OFF)
SET(USE_RLENVS_CPP ON)
SET(RLENVS_CPP_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/external/rlenvs_cpp/src")
SET(RLENVS_CPP_LIB_PATH "${PROJECT_SOURCE_DIR}/external/rlenvs_cpp/install/dbg")

# PyTorch
SET(USE_PYTORCH OFF)
SET(PYTORCH_PATH "/home/alex/MySoftware/libtorch")
	
SET(USE_CUDA OFF)

# The CUDA compiler
SET(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin")

# Which packages to use from the library
SET(USE_RL ON)
SET(DATA_SET_FOLDER ${PROJECT_SOURCE_DIR}/data)
SET(ENABLE_EXAMPLES_FLAG ON)
SET(ENABLE_TESTS_FLAG OFF) # Testing
SET(ENABLE_DOC_FLAG OFF) # Documentation
SET(BLAZE_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/external/blaze/install/include") # explicitly specify where the blaze c++ headers are

IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
	SET(CMAKE_INSTALL_PREFIX  ${PROJECT_SOURCE_DIR}/install/dbg)
	SET(CMAKE_CXX_FLAGS "-g ") # -Wall -Wextra")
	SET(CUBEAI_DEBUG ON)
	SET(CUBEAI_PRINT_DBG_MSGS ON)
ELSEIF(CMAKE_BUILD_TYPE STREQUAL "Release")
	SET(CMAKE_INSTALL_PREFIX  ${PROJECT_SOURCE_DIR}/install/opt)
	SET(CMAKE_CXX_FLAGS "-O2")
    SET(CUBEAI_DEBUG OFF)
ENDIF()



IF(USE_RLENVS_CPP)

	IF(CMAKE_BUILD_TYPE MATCHES "Debug")
		SET(RLENVS_CPP_LIB_DIRS  ${RLENVS_CPP_LIB_PATH})
	ELSE()
		SET(RLENVS_CPP_LIB_DIRS  ${RLENVS_CPP_LIB_PATH})
	ENDIF()
	
ENDIF()


IF(ENABLE_TESTS_FLAG)
	# if not manually populated we do a 
	# system scan in the CMakeLists of the tests directory
	SET(GTEST_INCLUDE_DIRS "/home/alex/MySoftware/gtest/install/include")
	SET(GTEST_BOTH_LIBRARIES "/home/alex/MySoftware/gtest/install/lib")
ENDIF()



# find Boost
FIND_PACKAGE(Boost 1.74.0 REQUIRED)
IF(Boost_FOUND)
	MESSAGE( STATUS  "Found needed Boost C++ library.")
	SET(Boost_USE_SHARED_LIBS ON)
ELSE()
	MESSAGE( FATAL_ERROR  "Boost C++ library is required but not found.")
ENDIF()

# find BLAS library
FIND_PACKAGE(BLAS REQUIRED)

IF(NOT BLAS_FOUND)
	MESSAGE( FATAL_ERROR  "Could not find OpenBLAS library.")
ELSE()
	MESSAGE( STATUS  "Found needed BLAS library.")
ENDIF()

# attempt to find blaze in the usr system
IF(BLAZE_INCLUDE_DIRS MATCHES " ")

	FIND_PACKAGE( blaze REQUIRED )
	
	IF(NOT blaze_FOUND)
		MESSAGE( FATAL_ERROR  "Could not find Blaze C++ library.")
	ELSE()
		
   		add_library( blaze_target INTERFACE )
   		target_link_libraries( blaze_target INTERFACE blaze::blaze )
		INCLUDE_DIRECTORIES(${BLAZE_INCLUDE_DIRS})
	ENDIF()
ELSE()
	INCLUDE_DIRECTORIES(${BLAZE_INCLUDE_DIRS})
ENDIF()

#IF(NLOHMANN_JSON_INCL_DIR MATCHES " ")

#	FIND_PACKAGE( nlohmann_json REQUIRED )
#	INCLUDE_DIRECTORIES(${INTERFACE_INCLUDE_DIRECTORIES})
#ELSE()
#	INCLUDE_DIRECTORIES(${NLOHMANN_JSON_INCL_DIR})

#ENDIF()

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# if using PyTorch append the path to libtorch
IF(USE_PYTORCH)
	LIST(APPEND CMAKE_PREFIX_PATH ${PYTORCH_PATH})
	FIND_PACKAGE(Torch REQUIRED CONFIG)
	MESSAGE(STATUS "TORCH Include directory ${TORCH_INCLUDE_DIRS}")
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
INCLUDE_DIRECTORIES(${RLENVS_CPP_INCLUDE_DIRS})

IF(USE_PYTORCH)
	INCLUDE_DIRECTORIES(${TORCH_INCLUDE_DIRS})
ENDIF()

INCLUDE_DIRECTORIES(include/)

# collect the source files for the library
FILE(GLOB SRCS src/cubeai/base/*.cpp
               src/cubeai/ml/*.cpp
			   src/cubeai/ml/*/*.cpp
               src/cubeai/rl/*.cpp
               src/cubeai/rl/*/*.cpp
               src/cubeai/io/*.cpp
               src/cubeai/planning/*.cpp
               src/cubeai/geom_primitives/*.cpp
               src/cubeai/utils/*.cpp
               src/cubeai/optimization/*.cpp
               src/cubeai/datasets/*.cpp
               src/cubeai/pytorch_utils/*.cpp)


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
