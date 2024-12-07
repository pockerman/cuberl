#include "cubeai/base/cubeai_types.h"
#include "cubeai/utils/iteration_counter.h"
#include "cubeai/maths/statistics/distributions/normal_dist.h"
#include "cubeai/maths/vector_math.h"
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
using cubeai::maths::stats::NormalDist;
using cubeai::utils::IterationCounter;

// we will sample from the normal distribution
// with mu = 4.0 and std = 1.0
const real_t MU = 4.0;
const real_t STD = 1.0;

// sample size we draw per iteration
const uint_t N = 100;

// how many iterations to run
const uint_t ITERATIONS = 1000;

const uint_t SEED = 42;


// simple function that computes the
// value of h at a given point
real_t h(real_t x){
    return 1.0 ? x > 3.0: 0.0;
}

}

int main() {

    using namespace intro_example_7;

    BOOST_LOG_TRIVIAL(info)<<"Starting example...";
	
	// simple object to control iterations
	IterationCounter counter(ITERATIONS);
	NormalDist dist(MU, STD);
	NormalDist proposal_dist(0.0, 1.0);
	
	std::vector<real_t> intergals;
	intergals.reserve(ITERATIONS);
	
	while(counter.continue_iterations()){
		
		real_t integral = 0.0;
		
		// sample from the distribution
		auto sample = dist.sample_many(N, SEED);
		
		// for every point in the sample compue
		// the PDF value
		for(auto p: sample){
			auto nom = h(p) * proposal_dist.pdf(p);
			auto denom = dist.pdf(p);
			auto val = nom / denom;
			integral += val;
		}
		
		intergals.push_back(integral / static_cast<real_t>(N));
		
	}
	
	auto E_I = cubeai::maths::mean(intergals.begin(), intergals.end(), true);
	auto V_I = cubeai::maths::variance(intergals.begin(), intergals.end(), true);
	BOOST_LOG_TRIVIAL(info)<<"E[I]="<<E_I;
	BOOST_LOG_TRIVIAL(info)<<"V[I]="<<V_I;
	BOOST_LOG_TRIVIAL(info)<<"Finished example...";
    return 0;
}

