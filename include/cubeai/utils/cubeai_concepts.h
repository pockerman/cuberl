#ifndef CUBEAI_CONCEPTS_H
#define CUBEAI_CONCEPTS_H
/**
  * Constraints used around the library to make code
  * safer and more readable. A nice introduction to
  * C++ concepts can be found at:
  *
  * https://www.youtube.com/watch?v=N_kPd2OK1L8
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

template<typename Type>
concept is_default_constructible = std::is_default_constructible<Type>::value;

}

}
}

#endif // CUBEAI_CONSTRAINTS_H
