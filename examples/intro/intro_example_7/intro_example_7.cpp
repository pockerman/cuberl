#include "cubeai/base/cubeai_types.h"
#include "cubeai/utils/iteration_counter.h"
#include "cubeai/maths/statistics/dist_sampler.h"
#include <boost/log/trivial.hpp>

#include <cmath>
#include <utility>
#include <tuple>
#include <iostream>
#include <random>
#include <algorithm>

namespace intro_example_7
{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynMat;
using cubeai::DynVec;
using cubeai::maths::stats::DistSampler;
using cubeai::utils::IterationCounter;

// we will sample from the normal distribution
// with mu = 4.0 and std = 1.0
const real_t MU = 4.0;
const real_t STD = 1.0;

// sample size we draw per iteration
const uint_t N = 100;

// how many iterations to run
const ITERATIONS = 1000;

const uint_t SEED = 42;


// simple function that computes the
// value of h at a given point
real_t h(real_t x){
    return 1.0 ? x > 3.0: 0.0;
}

}

int main() {

    using namespace intro_example_7;

    BOOST_LOG_TRIVIAL(info)<<"Starting integration...";
	
	DistSampler sampler(SEED);
	
	// simple object to control iterations
	IterationCounter counter(ITERATIONS);
	std::vector<real_t> sample(N, 0.0);
	std::normal_distribution<real_t> dist(MU, STD);
	while(counter.continue_iterations()){
		
		real_t integral = 0.0;
		
		sampler.sample(dist, sample);
		
	}

	BOOST_LOG_TRIVIAL(info)<<"Finished computation...";
    return 0;
}

