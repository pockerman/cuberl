#ifndef DIST_SAMPLER_H
#define DIST_SAMPLER_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/utils/iteration_counter.h"
#include <random>


namespace cubeai {
namespace maths {
namespace stats {

class DistSampler
{
public:
	
	///
	/// \brief Constructor
	///
	DistSampler()=default;
	
	///
	///
	/// \brief Constructor
	explicit DistSampler(uint_t seed);
	
	
	template<typename DistType>
	real_t sample(const DistType& dist)const;
	
	template<typename DistType, typename VecType>
	void sample(const DistType& dist, VecType& result)const;
	
private:
	
	std::mt19937 gen_;
	
}

inline
DistSampler::DistSampler(uint_t seed)
:
gen_(seed)
{}


template<typename DistType>
real_t 
DistSampler::sample(const DistType& dist)const{
	return dist(gen_);
}

template<typename DistType, typename VecType>
void 
DistSampler::sample(const DistType& dist, VecType& result)const{
	
	IterationCounter counter(result.size());
	while(counter.continue_iterations()){
		auto x = dist(gen);
		result[current_iteration_index()] = x;
	}
	
}

	
}
}
}