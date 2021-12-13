#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/algorithm_base.h"

#include "cubeai/rl/policies/epsilon_greedy_policy.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/mountain_car.h"
#include "gymfcpp/time_step.h"

#include <vector>
#include <iostream>


namespace example{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::algos::AlgorithmBase;
using gymfcpp::MountainCar;

typedef gymfcpp::TimeStep<uint_t> time_step_t;

template<typename ObsType, typename BinType>
std::vector<real_t>
get_aggregated_state(const ObsType& obs, const BinType& pos_bin, const BinType& vel_bin){

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

private:

    uint_t n_itrs_per_episode_;
    Env& env_;
    real_t lr_;
    real_t gamma_;
    real_t dt_{1.0};
};

template<typename Env>
ApproxMC<Env>::ApproxMC(Env& env, uint_t n_episodes, uint_t n_itrs_per_episode,
                        real_t tolerance, real_t lr, real_t gamma)
    :
     AlgorithmBase(n_episodes, tolerance),
     n_itrs_per_episode_(n_itrs_per_episode),
     env_(env),
     lr_(lr),
     gamma_(gamma)
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
    }
}

template<typename Env>
void
ApproxMC<Env>::step(){

    for(uint_t itr=0; itr<n_itrs_per_episode_; ++itr){

    }


}

}


int main(){

    using namespace example;

    try{

        const real_t GAMMA = 1.0;
        const uint_t N_EPISODES = 20000;
        const uint_t N_ITRS_PER_EPISODE = 2000;
        const real_t TOL = 1.0e-8;

        Py_Initialize();
        auto gym_module = boost::python::import("gym");
        auto gym_namespace = gym_module.attr("__dict__");

        MountainCar env("v0", gym_namespace, false);
        env.make();

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
