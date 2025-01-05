cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

SET(EXECUTABLE  intro_example_8)
SET(SOURCE ${EXECUTABLE}.cu)

ADD_EXECUTABLE(${EXECUTABLE} ${SOURCE})

# We need to explicitly state that we need all CUDA files
# to be built with -dc as the member functions 
# could be called by other libraries and executable
SET_TARGET_PROPERTIES(${EXECUTABLE} PROPERTIES CUDA_SEPARABLE_COMPLILATION ON)
			 
TARGET_LINK_LIBRARIES(${EXECUTABLE} cubeailib)
TARGET_LINK_LIBRARIES(${EXECUTABLE} rlenvscpplib)

IF( USE_PYTORCH )
    TARGET_LINK_LIBRARIES(${EXECUTABLE} ${TORCH_LIBRARIES})
ENDIF()

TARGET_LINK_LIBRARIES(${EXECUTABLE} pthread)
TARGET_LINK_LIBRARIES(${EXECUTABLE} openblas)
target_link_libraries(${EXECUTABLE} tbb)
target_link_libraries(${EXECUTABLE} boost_log)
