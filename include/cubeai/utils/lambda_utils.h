#ifndef LAMBDA_UTILS_H
#define LAMBDA_UTILS_H

#include "cubeai/base/cubeai_types.h"
#include <iostream>

namespace cubeai {
namespace utils {


///
/// \brief Utility lambda to print the given item.
/// Requires C++14
///
template<typename T>
auto cubeai_print = [](const T& val) { std::cout << val<<std::endl; };





}

}

#endif // ITERATION_COUNTER_H
