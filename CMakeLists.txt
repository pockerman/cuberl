CMAKE_MINIMUM_REQUIRED(VERSION 3.6 FATAL_ERROR)
MESSAGE(STATUS "Using CMake ${CMAKE_VERSION}")

PROJECT(cubeai VERSION 0.0.2 LANGUAGES CXX C)

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


SET(CUBEAILIB_VERSION_MAJOR 0)
SET(CUBEAILIB_VERSION_MINOR 0)
SET(CUBEAILIB_VERSION_PATCH 7)
SET(CUBEAILIB_VERSION "${CUBEAILIB_VERSION_MAJOR}.${CUBEAILIB_VERSION_MINOR}.${CUBEAILIB_VERSION_PATCH}-alpha")
MESSAGE(STATUS "CubeAI lib Version ${CUBEAILIB_VERSION}")

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_COMPILER g++-11)
SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_STANDARD_REQUIRED True)
SET(CMAKE_C_COMPILER gcc-11)
SET(CMAKE_C_STANDARD 20)
SET(CMAKE_C_STANDARD_REQUIRED True)


IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
SET(CMAKE_INSTALL_PREFIX  ${PROJECT_SOURCE_DIR}/install/dbg)
SET(CMAKE_CXX_FLAGS "-g ") # -Wall -Wextra")
ELSEIF(CMAKE_BUILD_TYPE STREQUAL "Release")
SET(CMAKE_INSTALL_PREFIX  ${PROJECT_SOURCE_DIR}/install/opt)
SET(CMAKE_CXX_FLAGS "-O2")
ENDIF()

SET(CMAKE_INSTALL_PREFIX /home/alex/qi3/cubeAI/install)
SET(CMAKE_LINKER_FLAGS "-pthread")

# explicitly specify where the blaze c++ headers are
SET(BLAZE_INCLUDE_DIRS "/home/alex/MySoftware/blaze-3.8/install/include")

# external JSON library
SET(NLOHMANN_JSON_INCL_DIR "/home/alex/MySoftware/nlohman_json/install/include")

# configuration for Python3
SET(PYTHON_CONFIG_LIBS_PATH "/usr/lib/python3.8/config-3.8-x86_64-linux-gnu/")

SET(USE_OPENMP ON)
SET(USE_LOG ON)
SET(USE_PYTORCH ON)
SET(USE_OPENCV OFF)
SET(USE_GYMFCPP ON)
SET(DATA_SET_FOLDER ${PROJECT_SOURCE_DIR}/data)

SET(ENABLE_EXAMPLES_FLAG ON)
SET(ENABLE_TESTS_FLAG ON)
SET(ENABLE_DOC_FLAG OFF)


IF(USE_GYMFCPP)
	SET(GYMFCPP_INCLUDE_DIRS "/home/alex/qi3/cubeAI/gymcpp/src")
	
	IF(CMAKE_BUILD_TYPE MATCHES "Debug")
		SET(GYMFCPP_LIB_DIRS  ${PROJECT_SOURCE_DIR}/gymcpp/install/dbg/)
	ELSE()
		SET(GYMFCPP_LIB_DIRS  ${PROJECT_SOURCE_DIR}gymcpp/install/opt/)
	ENDIF()
	
ENDIF()

IF(PYTHON_CONFIG_LIBS_PATH MATCHES " ")
	MESSAGE( FATAL_ERROR  "Python3 was not found.")
ENDIF()

IF(ENABLE_TESTS_FLAG)
	# if not manually populated we do a 
	# system scan in the CMakeLists of the tests directory
	SET(GTEST_INCLUDE_DIRS "/home/alex/MySoftware/gtest/install/include")
	SET(GTEST_BOTH_LIBRARIES "/home/alex/MySoftware/gtest/install/lib")
ENDIF()

IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
	SET(CUBEAI_DEBUG ON)
	SET(CUBEAI_PRINT_DBG_MSGS ON)
ENDIF()

# find Boost
FIND_PACKAGE(Boost 1.65.0 REQUIRED)
IF(Boost_FOUND)
	MESSAGE( STATUS  "Found needed Boost C++ library.")
	SET(Boost_USE_SHARED_LIBS ON)
ELSE()
	MESSAGE( FATAL_ERROR  "Boost C++ library is required but not found.")
ENDIF()


# find OpenBLAS
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
		INCLUDE_DIRECTORIES(${BLAZE_INCLUDE_DIRS})
	ENDIF()
ELSE()
	INCLUDE_DIRECTORIES(${BLAZE_INCLUDE_DIRS})
ENDIF()

IF(NLOHMANN_JSON_INCL_DIR MATCHES " ")

	FIND_PACKAGE( nlohmann_json REQUIRED )
	INCLUDE_DIRECTORIES(${INTERFACE_INCLUDE_DIRECTORIES})
ELSE()
	INCLUDE_DIRECTORIES(${NLOHMANN_JSON_INCL_DIR})

ENDIF()


LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

IF(USE_PYTORCH)
LIST(APPEND CMAKE_PREFIX_PATH /home/alex/MySoftware/libtorch)
FIND_PACKAGE(Torch REQUIRED CONFIG)
ENDIF()

MESSAGE(STATUS "TORCH Include directory ${TORCH_INCLUDE_DIRS}")
MESSAGE(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
MESSAGE(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
MESSAGE(STATUS "Project SRC directory: ${PROJECT_SOURCE_DIR}")

# write configuration and version files
configure_file(config.h.in ${PROJECT_SOURCE_DIR}/include/cubeai/base/cubeai_config.h @ONLY)
configure_file(version.h.in ${PROJECT_SOURCE_DIR}/include/cubeai/base/cubeai_version.h @ONLY)

# include directories
INCLUDE_DIRECTORIES(${BOOST_INCLUDEDIR})

IF(USE_PYTORCH)
INCLUDE_DIRECTORIES(${TORCH_INCLUDE_DIRS})
ENDIF()

INCLUDE_DIRECTORIES(include/)

# collect the source files for the library
FILE(GLOB SRCS src/cubeai/base/*.cpp src/cubeai/ml/*.cpp src/cubeai/ml/*/*.cpp src/cubeai/rl/*.cpp src/cubeai/rl/*/*.cpp
    src/cubeai/io/*.cpp src/cubeai/planning/*.cpp, src/cubeai/geom_primitives/*.cpp src/cubeai/utils/*.cpp src/cubeai/optimization/*.cpp
    src/cubeai/datasets/*.cpp src/cubeai/robots/*.cpp)


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
