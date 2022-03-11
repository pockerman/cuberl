#ifndef CUBEAI_CONCEPTS_H
#define CUBEAI_CONCEPTS_H
/**
  * Constraints used around the library to make code
  * safer and more readable
  * */

#include <type_traits>

namespace cubeai{
namespace utils {
namespace concepts {

template<typename VectorType>
concept integral_vector = std::is_integral<typename VectorType::value_type>::value;

template<typename VectorType>
concept float_vector = std::is_floating_point<typename VectorType::value_type>::value;

template<typename VectorType>
concept float_or_integral_vector = std::is_integral<typename VectorType::value_type>::value || std::is_floating_point<typename VectorType::value_type>::value;

}

}
}

#endif // CUBEAI_CONSTRAINTS_H
