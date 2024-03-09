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
#include <iterator>
#include <iostream>

namespace cubeai{
namespace maths{
namespace stats{

///
/// \brief mean Computes the mean value of the vector
///
template<utils::concepts::float_or_integral_vector VectorType>
real_t
mean(const VectorType& vector, bool parallel=true){

    auto sum = 0.0;

    if(parallel){
      sum   = std::reduce(std::execution::par, vector.begin(), vector.end(), sum);
    }
    else{
        sum = std::accumulate(vector.begin(), vector.end(), sum);
    }

    return sum / static_cast<real_t>(vector.size());
}

///
/// \brief mean computes the mean value of the given DynVec
///
template<typename T>
real_t
mean(const DynVec<T>& vector){
    return 0.0; //blaze::mean(vector);
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


template<utils::concepts::float_vector Vec>
typename Vec::value_type
choose_value(const Vec& vals, uint_t seed=42){

    std::mt19937 generator(seed);
    auto size = std::distance(vals.begin(), vals.end());
    std::discrete_distribution<int> distribution(0, static_cast<int>(size));
    auto rand_idx = distribution(generator);
    return vals[rand_idx];
}

///
/// \brief Given a vector of intergal or floating point values
/// and a set of values to choose from, randomize the entries ofv
///
template<utils::concepts::float_or_integral_vector Vec>
void
randomize_vec(Vec& v, const Vec& walk_set, uint_t seed=42){


    std::for_each(v.begin(), v.end(),
                  [&](auto& val){
                      val +=  choose_value(walk_set, seed);
                  });

}


}
}
}


#endif // BASIC_ARRAY_STATISTICS_H
