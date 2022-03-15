#ifndef BASIC_ARRAY_STATISTICS_H
#define BASIC_ARRAY_STATISTICS_H
/**
  * Implements basic statistical computations on arrays
  */

#include "cubeai/base/cubeai_types.h"
#include "cubeai/utils/cubeai_concepts.h"

#include <algorithm>
#include <execution>
#include <random>

namespace cubeai{
namespace maths{
namespace stats{

///
/// \brief Computes the sum of the vector elements
///
template<utils::concepts::float_or_integral_vector VectorType>
typename VectorType::value_type
sum(const VectorType& vector, bool parallel=true){

    auto sum_ = 0.0;

    if(parallel){
      sum_   = std::reduce(std::execution::par, vector.begin(), vector.end(), sum_);

    }
    else{
        sum_ = std::accumulate(vector.begin(), vector.end(), sum_);
    }
    return sum_;
}

///
/// \brief mean Computes the mean value of the vector
///
template<utils::concepts::float_or_integral_vector VectorType>
real_t
mean(const VectorType& vector, bool parallel=true){
    return sum(vector, parallel) / static_cast<real_t>(vector.size());
}

///
/// \brief mean computes the mean value of the given DynVec
///
template<typename T>
real_t
mean(const DynVec<T>& vector){
    return blaze::mean(vector);
}

///
///
///
template<utils::concepts::float_or_integral_vector VectorType>
typename VectorType::value_type
var(const VectorType& vector, real_t mu, bool dofs=true){

    // compute sum of squares
    auto vbegin = vector.begin();
    auto vend = vector.end();

    // Now calculate the variance
    auto variance_func = [&mu](typename VectorType::value_type accumulator, typename VectorType::value_type val) {
            return accumulator + (val - mu)*(val - mu);
    };

    auto var_ = std::accumulate(vector.begin(), vector.end(), 0.0, variance_func);

    /*auto sum_ = 0.0;
    while (vbegin != vend) {
        auto dif = *vbegin - mu;
        sum_ += dif * dif;
    }*/

    if(dofs){
        return var_ / (vector.size() - 1);
    }

    return  var_ / vector.size();
}

///
/// \brief choice. Implements similar functionality
/// to numpy.choice function
///
/// https://www.cplusplus.com/reference/random/discrete_distribution/
///
template<utils::concepts::float_vector Vec2>
uint_t
choice(const Vec2& probs, uint_t seed=42){

    std::mt19937 generator(seed);
    std::discrete_distribution<int> distribution(probs.begin(), probs.end());
    return distribution(generator);
}

///
/// \brief choice. Implements similar functionality
/// to numpy.choice function
///
/// https://www.cplusplus.com/reference/random/discrete_distribution/
///
template<utils::concepts::integral_vector Vec1, utils::concepts::float_vector Vec2>
uint_t
choice(const Vec1& choices, const Vec2& probs, uint_t seed=42){

    std::mt19937 generator(seed);
    std::discrete_distribution<int> distribution(probs.begin(), probs.end());
    return choices[distribution(generator)];
}


}
}
}


#endif // BASIC_ARRAY_STATISTICS_H
