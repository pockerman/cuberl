#include "cubeai/base/cubeai_types.h"


#include <cmath>
#include <utility>
#include <tuple>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>

namespace exe
{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynVec;

real_t get_reward(real_t prob, uint_t n=10){

    auto reward = 1;

    for(uint_t i=0; i<n; ++i){
        auto prob_value = static_cast <real_t> (rand()) / static_cast <real_t> (RAND_MAX);
        if (prob_value < prob){
            reward += 1;
        }
    }

    return reward;
}


real_t soft_max(const DynVec<real_t>& values, real_t tau=1.12){


   auto exp  = blaze::pow(values / tau);
   auto sum  = blaze::sum(exp);

    return exp / sum;
}

void update_record(std::vector<std::vector<real_t>>& records, uint_t action, real_t r){


    auto new_r = (records[action][0] * records[action][1] + r) / (records[action][0] + 1);
    records[action][0] += 1;
    records[action][1] = new_r;
}

uint_t get_best_arm(const std::vector<std::vector<real_t>>& records){

    std::vector<real_t> values(records.size(), 0.0);
    for(uint_t i=0; i<records.size(); ++i){
        values[i] = records[i][1];
    }

    auto iterator_result = std::max_element(values.begin(), values.end());
    return std::distance(values.begin(), iterator_result);
}

std::vector<real_t> get_probs(uint_t n){

    std::vector<real_t> probs(n);

    for(uint_t i=0; i<n; ++i){
        probs[i] = static_cast <real_t> (rand()) / static_cast <real_t> (RAND_MAX);
    }

    return probs;
}
}

int main() {

    using namespace exe;

    // this should return something close
    // to 7
    const uint_t N = 10;

    const auto N_EXPERIMENTS = 500;
    auto probs = get_probs(N);
    auto eps = 0.2;

    //Will be used to obtain a seed for the random number engine
    std::random_device rd;

    //Standard mersenne_twister_engine seeded with rd()
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 9);

    std::vector<std::vector<real_t>> records(N);

    for(uint_t i=0; i<N; ++i){
        records.resize(2);
    }

    std::vector<real_t> rewards;

    for(uint_t i=0; i<N_EXPERIMENTS; ++i){
        auto prob_val = static_cast <real_t> (rand()) / static_cast <real_t> (RAND_MAX);

        auto choice = distrib(gen);
        if(prob_val > eps){
            choice = get_best_arm(records);
        }

        auto r = get_reward(probs[choice]);
        update_record(records, choice, r);

        auto mean_reward = ( (i + 1) * rewards.back() + r) / (i + 2);
        rewards.push_back(mean_reward);
    }

   return 0;
}


