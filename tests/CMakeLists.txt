SET(CMAKE_CXX_COMPILER /usr/bin/g++-11)
SET(CMAKE_C_COMPILER /usr/bin/gcc-10)
SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_STANDARD_REQUIRED True) 

SET(CMAKE_CXX_FLAGS "-g -Wall -Wextra")
SET(CMAKE_LINKER_FLAGS "-pthread")

INCLUDE_DIRECTORIES(/home/alex/MySoftware/gtest/install/include)

# use the Boost link directories
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
LINK_DIRECTORIES(${CMAKE_INSTALL_PREFIX})
LINK_DIRECTORIES(/usr/lib/python3.8/config-3.8-x86_64-linux-gnu/)
LINK_DIRECTORIES(/home/alex/MySoftware/gtest/install/lib)

# add the test dirs
ADD_SUBDIRECTORY(test_array_utils)
ADD_SUBDIRECTORY(test_mc_tree_search)
