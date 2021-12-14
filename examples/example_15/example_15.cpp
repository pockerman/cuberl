#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/algorithm_base.h"
#include "cubeai/rl/policies/epsilon_greedy_policy.h"
#include "cubeai/rl/utilities/experience_buffer.h"
#include "cubeai/utils/numpy_utils.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/mountain_car.h"
#include "gymfcpp/time_step.h"

#include <vector>
#include <iostream>
#include <unordered_map>
#include <utility>


namespace example{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::algos::AlgorithmBase;
using cubeai::rl::ExperienceBuffer;
using gymfcpp::MountainCar;

typedef gymfcpp::TimeStep<uint_t> time_step_t;

const real_t GAMMA = 1.0;
const uint_t N_EPISODES = 20000;
const uint_t N_ITRS_PER_EPISODE = 2000;
const real_t TOL = 1.0e-8;

auto pos_bins = std::vector<real_t>();
auto vel_bins = std::vector<real_t>();

struct Transition{};

struct Policy
{

    uint_t operator()(real_t vel)const{

        if(vel < 0.0){
            return 0;
        }

        return 2;
    }
};

template<typename ObsType, typename BinType>
std::pair<real_t, real_t>
get_aggregated_state(const ObsType& obs, const BinType& pos_bin, const BinType& vel_bin){

    return {0.0, 0.0};
}


template<typename Env>
class ApproxMC: public AlgorithmBase
{
public:

    ApproxMC(Env& env, uint_t n_episodes, uint_t n_itrs_per_episode,
             real_t tolerance, real_t lr, real_t gamma);

    virtual void actions_before_training_iterations();
    virtual void actions_after_training_iterations() final override {}
    virtual void actions_before_training_episode() final override;
    virtual void actions_after_training_episode() final override{}
    virtual void step() final override;

    ///
    /// \brief reset. Reset the underlying data structures to the point when the constructor is called.
    ///
    virtual void reset();


    real_t state_value(real_t pos, real_t vel)const;

    void update_weights(real_t total_return, std::pair<real_t, real_t> state, real_t t);

private:

    uint_t n_itrs_per_episode_;
    Env& env_;
    real_t lr_;
    real_t gamma_;
    real_t dt_{1.0};

    std::unordered_map<std::pair<real_t, real_t>, real_t> weights_;

    std::vector<real_t> near_exit_;
    std::vector<real_t> left_side_;

    ExperienceBuffer<Transition> memory_;

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
ApproxMC<Env>::actions_before_training_iterations(){
    this->AlgorithmBase::reset();
    env_.reset();
}

template<typename Env>
void
ApproxMC<Env>::actions_before_training_episode(){

    if(this->current_iteration() % 1000 ==0){
        dt_ += 0.1;

        auto idx = this->current_iteration() /  1000;
        auto state = get_aggregated_state((0.43, 0.054), pos_bins, vel_bins);
        near_exit_[idx] = state_value(state.first, state.second);
        state = get_aggregated_state((-1.1, 0.001), pos_bins, vel_bins);
        left_side_[idx] = state_value(state.first, state.second);
    }

    // clear any items in the memory
    memory_.clear();
}

template<typename Env>
real_t
ApproxMC<Env>::state_value(real_t pos, real_t vel)const{

    return 0.0; //weights_[std::make_pair(pos, vel)];
}

template<typename Env>
void
ApproxMC<Env>::update_weights(real_t total_return, std::pair<real_t, real_t> state, real_t t){


}

template<typename Env>
void
ApproxMC<Env>::step(){

    auto env_state = env_.reset();

    for(uint_t itr=0; itr<n_itrs_per_episode_; ++itr){

        auto state = get_aggregated_state(env_state.observation(), pos_bins, vel_bins);
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

        // generate position and velocity bins
        pos_bins = cubeai::numpy_utils::linespace(-1.2, 0.5, 8);
        vel_bins = cubeai::numpy_utils::linespace(-0.07, 0.07, 8);

        std::vector<real_t> lrs{0.1, 0.01, 0.001};

        for(const auto lr: lrs){

            auto model = ApproxMC<MountainCar>(env, N_EPISODES, N_ITRS_PER_EPISODE,
                                               TOL, lr, GAMMA);
            model.train();
        }





    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    }
    catch(...){

        std::cout<<"Unknown exception occured"<<std::endl;
    }
    return 0;
}
