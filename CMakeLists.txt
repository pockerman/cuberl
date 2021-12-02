CMAKE_MINIMUM_REQUIRED(VERSION 3.6 FATAL_ERROR)
MESSAGE(STATUS "Using CMake ${CMAKE_VERSION}")

PROJECT(cubeai VERSION 0.0.1 LANGUAGES CXX C)

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
SET(CUBEAILIB_VERSION_PATCH 1)
SET(CUBEAILIB_VERSION "${CUBEAILIB_VERSION_MAJOR}.${CUBEAILIB_VERSION_MINOR}.${CUBEAILIB_VERSION_PATCH}")
MESSAGE(STATUS "CubeAI lib Version ${CUBEAILIB_VERSION}")

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_COMPILER gcc)
SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_STANDARD_REQUIRED True)
SET(CMAKE_C_COMPILER gcc)
SET(CMAKE_INSTALL_PREFIX /home/alex/qi3/cubeAI/install)
SET(CMAKE_LINKER_FLAGS "-pthread")

SET(BLAZE_INCL_DIR "/home/alex/MySoftware/blaze-3.8/install/include")
SET(NLOHMANN_JSON_INCL_DIR "/home/alex/MySoftware/nlohman_json/install/include")
SET(MAGIC_ENUM_INCL_DIR " ")

SET(USE_OPENMP ON)
SET(USE_LOG ON)
SET(USE_PYTORCH ON)
SET(USE_OPENCV OFF)
SET(USE_GYMFCPP ON)
SET(ENABLE_EXAMPLES_FLAG ON)
SET(ENABLE_TESTS_FLAG OFF)

IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
SET(CUBEAI_DEBUG ON)
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

LIST(APPEND CMAKE_PREFIX_PATH /home/alex/MySoftware/libtorch)
FIND_PACKAGE(Torch REQUIRED CONFIG)

MESSAGE(STATUS "TORCH Include directory ${TORCH_INCLUDE_DIRS}")
MESSAGE(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
MESSAGE(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
MESSAGE(STATUS "Project SRC directory: ${PROJECT_SOURCE_DIR}")

configure_file(config.h.in ${PROJECT_SOURCE_DIR}/include/cubeai/base/cubeai_config.h @ONLY)
configure_file(version.h.in ${PROJECT_SOURCE_DIR}/include/cubeai/base/cubeai_version.h @ONLY)

INCLUDE_DIRECTORIES(/home/alex/MySoftware/blaze-3.8/install/include)
INCLUDE_DIRECTORIES(${BOOST_INCLUDEDIR})
INCLUDE_DIRECTORIES(/home/alex/MySoftware/nlohman_json/install/include)
INCLUDE_DIRECTORIES(${BLAZE_INCL_DIR})
INCLUDE_DIRECTORIES(${BOOST_INCLUDEDIR})
INCLUDE_DIRECTORIES(${TORCH_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(include/)

FILE(GLOB SRCS src/cubeai/base/*.cpp src/cubeai/ml/*.cpp src/cubeai/ml/*/*.cpp src/cubeai/rl/*.cpp src/cubeai/rl/*/*.cpp)

ADD_LIBRARY(cubeailib SHARED ${SRCS})
SET_TARGET_PROPERTIES(cubeailib PROPERTIES LINKER_LANGUAGE CXX)
INSTALL(TARGETS cubeailib DESTINATION ${CMAKE_INSTALL_PREFIX})
MESSAGE(STATUS "Installation destination at: ${CMAKE_INSTALL_PREFIX}")

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
MESSAGE(STATUS "Type make install to build cubeAI")
