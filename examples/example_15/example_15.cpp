#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/algorithm_base.h"
#include "cubeai/rl/policies/epsilon_greedy_policy.h"
#include "cubeai/rl/utilities/vector_experience_buffer.h"
#include "cubeai/utils/array_utils.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/mountain_car.h"
#include "gymfcpp/time_step.h"

#include <boost/python.hpp>

#include <vector>
#include <iostream>
#include <unordered_map>
#include <utility>


namespace example{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::algos::AlgorithmBase;
using cubeai::rl::VectorExperienceBuffer;
using gymfcpp::MountainCar;

typedef gymfcpp::TimeStep<uint_t> time_step_t;
typedef std::pair<uint_t, uint_t> state_type;

const real_t GAMMA = 1.0;
const uint_t N_EPISODES = 20000;
const uint_t N_ITRS_PER_EPISODE = 2000;
const real_t TOL = 1.0e-8;

// generate position and velocity bins
auto pos_bins = std::vector<real_t>({-1.2, -0.95714286, -0.71428571, -0.47142857, -0.22857143, 0.01428571,  0.25714286,  0.5});
auto vel_bins = std::vector<real_t>({-0.07, -0.05, -0.03, -0.01,  0.01,  0.03,  0.05,  0.07});

template<typename ItrTp, typename StateTp>
ItrTp is_state_included(ItrTp begin, ItrTp end, const StateTp& state){
    return begin;
}

struct Transition
{

    state_type state;
    real_t reward;
    uint_t action;
    bool done;
};

struct Policy
{

    uint_t operator()(real_t vel)const{

        if(vel < 0.0){
            return 0;
        }

        return 2;
    }
};

template<typename BinType>
std::pair<uint_t, uint_t>
get_aggregated_state(const std::pair<real_t, real_t>& obs, const BinType& pos_bins,
                     const BinType& vel_bins){

    auto pos = cubeai::bin_index(obs.first, pos_bins);
    auto vel = cubeai::bin_index(obs.second, vel_bins);
    return {pos, vel};
}


template<typename Env>
class ApproxMC: public AlgorithmBase
{
public:

    ApproxMC(Env& env, uint_t n_episodes, uint_t n_itrs_per_episode,
             real_t tolerance, real_t lr, real_t gamma);

    virtual void actions_before_training_episodes() final override;
    virtual void actions_after_training_episodes() final override {}
    virtual void actions_before_training_episode() final override;
    virtual void actions_after_training_episode() final override;
    virtual void on_episode() final override;
    virtual void reset();
    real_t state_value(uint_t pos, uint_t vel)const;
    void update_weights(real_t total_return, state_type state, real_t t);

private:

    uint_t n_itrs_per_episode_;
    Env& env_;
    real_t lr_;
    real_t gamma_;
    real_t dt_{1.0};

    //std::vector<std::tuple<state_type, real_t>> weights_;
    std::map<std::pair<uint_t, uint_t>, real_t> weights_;

    std::vector<real_t> near_exit_;
    std::vector<real_t> left_side_;

    VectorExperienceBuffer<Transition> memory_;

    Policy policy_;
};

template<typename Env>
ApproxMC<Env>::ApproxMC(Env& env, uint_t n_episodes, uint_t n_itrs_per_episode,
                        real_t tolerance, real_t lr, real_t gamma)
    :
     AlgorithmBase(n_episodes, tolerance),
     n_itrs_per_episode_(n_itrs_per_episode),
     env_(env),
     lr_(lr),
     gamma_(gamma),
     near_exit_(N_EPISODES / 1000, 0.0),
     left_side_(N_EPISODES / 1000, 0.0)

{}

template<typename Env>
void
ApproxMC<Env>::actions_before_training_episodes(){
   this->reset();
}

template<typename Env>
void
ApproxMC<Env>::reset(){

    this->AlgorithmBase::reset();
    env_.reset();
    dt_ = 1.0;
}

template<typename Env>
void
ApproxMC<Env>::actions_before_training_episode(){

    if(this->current_episode_idx() % 1000 ==0){
        dt_ += 0.1;

        auto idx = this->current_episode_idx() /  1000;
        auto state = get_aggregated_state(std::make_pair(0.43, 0.054), pos_bins, vel_bins);
        near_exit_[idx] = state_value(state.first, state.second);

        state = get_aggregated_state(std::make_pair(-1.1, 0.001), pos_bins, vel_bins);
        left_side_[idx] = state_value(state.first, state.second);
    }

    // clear any items in the memory
    memory_.clear();
}

template<typename Env>
void
ApproxMC<Env>::actions_after_training_episode(){

    auto last = true;
    auto G = 0.0;
    std::vector<std::tuple<std::pair<uint_t, uint_t>, real_t>> states_returns;

    auto rbegin = memory_.rbegin();
    auto rend   = memory_.rend();

    for(; rbegin != rend; ++rbegin){

        const auto& experience = *rbegin;

        if(last){
            last = false;
        }
        else{
            states_returns.push_back(std::make_tuple(experience.state, G));
        }

         G *= gamma_  + experience.reward;
    }

    auto r_state_begin = states_returns.rbegin();
    auto r_state_end = states_returns.rend();

    std::vector<state_type> states_visited;
    states_visited.reserve(states_returns.size());

    for(; r_state_begin != r_state_end; ++r_state_begin){

        const auto& state = std::get<0>(*r_state_begin);
        const auto G = std::get<1>(*r_state_begin);

        if(is_state_included(states_visited.begin(), states_visited.end(), state) == states_visited.end()){
             update_weights(G, state, dt_);
             states_visited.push_back(state);
        }
    }
}

template<typename Env>
real_t
ApproxMC<Env>::state_value(uint_t pos, uint_t vel)const{

    auto itr = weights_.find(std::make_pair(pos, vel));
    return itr->second;
}

template<typename Env>
void
ApproxMC<Env>::update_weights(real_t total_return, state_type state, real_t t){

    auto itr = weights_.find(state);
    if( itr != weights_.end()){

        std::get<1>(*itr) = total_return;
    }
    else{
        weights_.insert({state, total_return});
    }
}

template<typename Env>
void
ApproxMC<Env>::on_episode(){

    auto env_state = env_.reset();

    for(uint_t itr=0; itr<n_itrs_per_episode_; ++itr){

        auto state = get_aggregated_state({env_state.observation()[0], env_state.observation()[1]} , pos_bins, vel_bins);
        auto action = policy_(state.second);

        auto next_time_step = env_.step(action);
    }
}

}


int main(){

    using namespace example;

    try{

        Py_Initialize();
        auto gym_module = boost::python::import("gym");
        auto gym_namespace = gym_module.attr("__dict__");

        MountainCar env("v0", gym_namespace, false);
        env.make();

        std::vector<real_t> lrs{0.1, 0.01, 0.001};

        for(const auto lr: lrs){

            auto model = ApproxMC<MountainCar>(env, N_EPISODES,
                                               N_ITRS_PER_EPISODE, TOL, lr, GAMMA);
            model.do_verbose_output();
            model.train();
        }

    }
    catch(const boost::python::error_already_set&)
    {
            PyErr_Print();
    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    }
    catch(...){

        std::cout<<"Unknown exception occured"<<std::endl;
    }
    return 0;
}
