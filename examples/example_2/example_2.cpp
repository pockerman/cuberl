/**
  * Solve the multi-arm bandit problem using
  * epsilon-greedy policy.  When using an epsilon-greedy policy
  * we choose an acion at random with probability  ε. Whilst with probability 1 – ε we will choose
  * the best lever based on what we currently know from past plays.
  * Most of the time we will play greedy, but sometimes we will take a risk and choose a random lever to see what happens.
  * The result will, of course, influence our future greedy actions.
  * For this example we will solve a 10-armed bandit problem, so N=10.
  *
  * This example is taken from the book: Reinforcement Learning in Action by Manning Publications.
  *
  * */

#include "cubeai/base/cubeai_types.h"
#include "cubeai/utils/array_utils.h"

#include <cmath>
#include <utility>
#include <tuple>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>
#include <utility>

namespace example2
{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynMat;
using cubeai::DynVec;


//  Each arm has an associated probability that weights how much reward it pays out.
// The reward probability distributions is implemented as follows.
// Each arm will have a probability (see the get_probs function below), e.g., 0.7.
// The maximum reward is $10.
// We will set up a for loop going to N (in this example N=10),
// and at each step it will add 1 to the reward if a random float is less than the arm’s probability.
// For example, assume that the probabiity given is 0.7 on the first loop it makes up a random float (e.g., 0.4). 0.4
// is less than 0.7, so reward += 1. On the next iteration, it makes up another
// random float (e.g., 0.6) which is also less than 0.7, so reward += 1.
// This continues until we complete 10 iterations, and then we return the
// final total reward, which could be anything between 0 and 10.
// With an arm probability of 0.7, the average reward of doing this to infinity would be 7,
// but on any single play it could be more or less.

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

// Helper function to tract the reward (actually the mean reward) for
// each arm. The records double array hold
// The first column of the records array will store the number of times each arm has been pulled,
// and the second column will store the running average reward.
void update_record(std::vector<std::pair<uint_t, real_t>>& records, uint_t action, real_t r){

    auto new_r = (records[action].first * records[action].second + r) / (records[action].first + 1);
    records[action].first += 1;
    records[action].second = new_r;
}


// Returns the index of the best arm by choosing
// the arm associated with the highest average reward,
uint_t get_best_arm(const std::vector<std::pair<uint_t, real_t>>& records){

    std::vector<real_t> values(records.size(), 0.0);
    for(uint_t i=0; i<records.size(); ++i){
        values[i] = records[i].second;
    }

    return cubeai::arg_max(values);

    //auto iterator_result = std::max_element(values.begin(), values.end());
    //return std::distance(values.begin(), iterator_result);
}

// Generate the probabilities associated with each
// of the n arm-bandits
std::vector<real_t> get_probs(uint_t n){

    std::vector<real_t> probs(n);

    for(uint_t i=0; i<n; ++i){
        probs[i] = static_cast <real_t> (rand()) / static_cast <real_t> (RAND_MAX);
    }

    return probs;
}
}

int main() {

    using namespace example2;

    // this should return something close
    // to 7
    const uint_t N = 2000;
    std::vector<real_t> values(N, 0.0);

    for(uint_t i=0; i<N; ++i){
        values[i] = get_reward(0.7);
    }

    auto mean = std::accumulate(values.begin(), values.end(), 0.0);

    std::cout<<"Mean is"<<mean<<std::endl;

    // Number of arm-bandits or total umber
    // of actions to choose from
    const auto n = 10;

    // number of games we want to play
    const auto N_GAMES = 500;

    // epsilon constant
    const auto EPS = 0.2;

    // Each arm has an associated probability
    // that weights how much reward it pays out.
    auto probs = get_probs(n);

    //Will be used to obtain a seed for the random number engine
    std::random_device rd;

    //Standard mersenne_twister_engine seeded with rd()
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 9);

    std::vector<std::pair<uint_t,real_t>> records(n);

    for(uint_t i=0; i<n; ++i){
        records[i].first = 0;
        records[i].second = 0.0;
    }

    // the mean rewards received
    std::vector<real_t> rewards;

    for(uint_t i=0; i<N_GAMES; ++i){

        // generate a random number
        auto prob_val = static_cast <real_t> (rand()) / static_cast <real_t> (RAND_MAX);

        // randomly select an action
        auto choice = distrib(gen);

        //...but if the generated random number is
        // greater than epsilon be greedy and select
        // the best arm that we currently have
        if(prob_val > EPS){
            choice = get_best_arm(records);
        }

        auto r = get_reward(probs[choice]);
        update_record(records, choice, r);

        auto mean_reward = ( (i + 1) * rewards.back() + r) / (i + 2);
        rewards.push_back(mean_reward);
    }

   return 0;
}


