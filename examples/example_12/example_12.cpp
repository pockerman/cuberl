#include "cubeai/base/cubeai_types.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/cart_pole.h"
#include "gymfcpp/time_step.h"

#include <torch/torch.h>

#include <deque>
#include <iostream>


namespace example{

using cubeai::real_t;
using cubeai::uint_t;

// Environment wrapper
class Environment
{
public:



};


//
class CartPoleDQNImpl: public torch::nn::Module
{
public:

    //
    CartPoleDQNImpl(uint_t h, uint_t w, uint_t output);

    // forward
    torch::Tensor forward(torch::Tensor input);

private:


    torch::nn::Conv2d conv1_;
    torch::nn::BatchNorm2d bn1_;
    torch::nn::Conv2d conv2_;
    torch::nn::BatchNorm2d bn2_;
    torch::nn::Conv2d conv3_;
    torch::nn::BatchNorm2d bn3_;
    torch::nn::Linear linear_;

    // helper function
    uint_t conv2d_size_out_(uint_t size, uint_t kernel_size=5, uint_t stride=2)const{return (size - (kernel_size - 1) - 1) / stride +1;}
    uint_t get_linear_input_size_(uint_t h, uint_t w) const;

};


CartPoleDQNImpl::CartPoleDQNImpl(uint_t h, uint_t w, uint_t output_size)
    :
     torch::nn::Module(),
     conv1_(torch::nn::Conv2dOptions(3, 16, 5).stride(2)),
     bn1_(16),
     conv2_(torch::nn::Conv2dOptions(16, 32, 5).stride(2)),
     bn2_(32),
     conv3_(torch::nn::Conv2dOptions(32, 32, 5).stride(2)),
     bn3_(32),
     linear_(torch::nn::Linear(get_linear_input_size_(h, w), output_size))


{

    // register_module() is needed if we want to use the parameters() method later on
    register_module("linear", linear_);
    register_module("conv1", conv1_);
    register_module("conv2", conv2_);
    register_module("conv3", conv3_);
    register_module("batch_norm1", bn1_);
    register_module("batch_norm2", bn2_);
    register_module("batch_norm3", bn2_);

}

uint_t
CartPoleDQNImpl::get_linear_input_size_(uint_t h, uint_t w) const{

    auto convw = conv2d_size_out_(conv2d_size_out_(conv2d_size_out_(w)));
    auto convh = conv2d_size_out_(conv2d_size_out_(conv2d_size_out_(h)));
    auto linear_input_size = convw * convh * 32;
    return linear_input_size;
}


}


int main(){

    using namespace example;

    try{

        /*Py_Initialize();
        auto gym_module = boost::python::import("gym");
        auto gym_namespace = gym_module.attr("__dict__");

        CliffWalkingEnv env(gym_namespace);
        env.build(true);

        std::cout<<"Number of states="<<env.n_states()<<std::endl;
        std::cout<<"Number of actions="<<env.n_actions()<<std::endl;

        EpsilonGreedyPolicy policy(0.005, env.n_actions(), EpsilonDecayOption::NONE);
        ExpectedSARSA<gymfcpp::TimeStep, EpsilonGreedyPolicy> expected_sarsa(5000, 1.0e-8,
                                                                             1.0, 0.01, 100, env, 1000, policy);

        expected_sarsa.do_verbose_output();

        std::cout<<"Starting training..."<<std::endl;
        auto train_result = expected_sarsa.train();

        std::cout<<train_result<<std::endl;
        std::cout<<"Finished training..."<<std::endl;

        std::cout<<"Saving value function..."<<std::endl;
        std::cout<<"Value function..."<<expected_sarsa.value_func()<<std::endl;
        expected_sarsa.save("expected_sarsa_value_func.csv");
        expected_sarsa.save_avg_scores("expected_sarsa_avg_scores.csv");
        expected_sarsa.save_state_action_function("expected_sarsa_state_action_function.csv");*/

    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    }
    catch(...){

        std::cout<<"Unknown exception occured"<<std::endl;
    }
    return 0;
}
