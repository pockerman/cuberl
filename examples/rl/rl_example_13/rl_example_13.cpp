/**
 * This example illustrates how to use the REINFORCE algorithm
 * on the CartPole environment from Gymnasium
 **/
#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"

#include "cubeai/rl/algorithms/pg/simple_reinforce.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/maths/optimization/optimizer_type.h"
#include "cubeai/maths/optimization/pytorch_optimizer_factory.h"
#include "cubeai/maths/statistics/distributions/torch_categorical.h"

#include "rlenvs/utils/io/csv_file_writer.h"
#include "rlenvs/envs/api_server/apiserver.h"
#include "rlenvs/envs/gymnasium/classic_control/cart_pole_env.h"

#include <boost/log/trivial.hpp>

#include <torch/torch.h>

#include <unordered_map>
#include <iostream>
#include <string>
#include <any>
#include <filesystem>
#include <map>

namespace rl_example_13{


const std::string SERVER_URL = "http://0.0.0.0:8001/api";
const std::string EXPERIMENT_ID = "2";

namespace F = torch::nn::functional;

using cuberl::real_t;
using cuberl::uint_t;
using cuberl::torch_tensor_t;
using cuberl::DeviceType;
using cuberl::rl::algos::pg::ReinforceSolver;
using cuberl::rl::algos::pg::ReinforceConfig;
using cuberl::rl::RLSerialAgentTrainer;
using cuberl::rl::RLSerialTrainerConfig;
using cuberl::maths::stats::TorchCategorical;
using rlenvscpp::envs::gymnasium::CartPole;


const uint_t L1 = 4;
const uint_t L2 = 150;
const uint_t L3 = 3;
const real_t LEARNING_RATE = 0.0009;


// The class that models the Policy network to train
class PolicyNetImpl: public torch::nn::Module
{
public:

    PolicyNetImpl();
    torch_tensor_t forward(torch_tensor_t state);

    template<typename StateTp>
    std::tuple<uint_t, real_t> act(const StateTp& state);
	
private:

   torch::nn::Linear fc1_;
   torch::nn::Linear fc2_;
};


PolicyNetImpl::PolicyNetImpl()
    :
      fc1_(torch::nn::Linear(4, 16)),
      fc2_(torch::nn::Linear(16, 2))
{
    register_module("fc1", fc1_);
    register_module("fc2", fc2_);
}


torch_tensor_t
PolicyNetImpl::forward(torch_tensor_t x){

    x = F::relu(fc1_->forward(x));
    x = fc2_->forward(x);
    return F::softmax(x, F::SoftmaxFuncOptions(0));
}


template<typename StateTp>
std::tuple<uint_t, real_t>
PolicyNetImpl::act(const StateTp& state){

    auto torch_state = torch::tensor(state);

    auto probs = forward(torch_state);
    auto m = TorchCategorical(probs, false);
    auto action = m.sample();
    return std::make_tuple(action.item().toLong(), 
	                       m.log_prob(action).item().to<real_t>());

}

TORCH_MODULE(PolicyNet);


struct Loss_1
{
	torch_tensor_t operator()(torch_tensor_t preds, torch_tensor_t y)const;
};

torch_tensor_t 
Loss_1::operator()(torch_tensor_t preds, torch_tensor_t y)const{
	return -1.0 * torch::sum(y * torch::log(preds));
}

typedef Loss_1 loss_type;
typedef CartPole env_type;
typedef PolicyNet policy_type;
typedef ReinforceSolver<env_type, PolicyNet, loss_type> solver_type;
}


int main(){

	BOOST_LOG_TRIVIAL(info)<<"Starting agent training";
    using namespace rl_example_13;

    try{

        // let's create a directory where we want to
        //store all the results from running a simualtion
        std::filesystem::create_directories("experiments/" + EXPERIMENT_ID);
        torch::manual_seed(42);

		BOOST_LOG_TRIVIAL(info)<<"Creating environment...";
		
		
		RESTApiServerWrapper server(SERVER_URL, true);
		
        auto env = CartPole(server);
	
        std::unordered_map<std::string, std::any> options;

        // with Gymnasium v0 is not working
        env.make("v1", options);
        env.reset();

        BOOST_LOG_TRIVIAL(info)<<"Done...";
		BOOST_LOG_TRIVIAL(info)<<"Number of actions="<<env.n_actions();

        PolicyNet policy;

        
        // reinforce options
        ReinforceConfig opts = {true, 1000, 100, 100, 
		                        100, 1.0e-2, 0.1, 195.0,
								DeviceType::CPU};

        std::map<std::string, std::any> opt_options;
        opt_options.insert(std::make_pair("lr", LEARNING_RATE));

		using namespace cuberl::maths;
        auto pytorch_ops = optim::pytorch::build_pytorch_optimizer_options(optim::OptimzerType::ADAM,
																		   opt_options);

        auto policy_optimizer = optim::pytorch::build_pytorch_optimizer(optim::OptimzerType::ADAM,
																		*policy, pytorch_ops);

		loss_type loss;
        solver_type solver(opts, policy, loss, policy_optimizer);


        RLSerialTrainerConfig config;
        config.n_episodes = 100;
        config.output_msg_frequency = 10;
        RLSerialAgentTrainer<env_type, solver_type> trainer(config, solver);
        trainer.train(env);

        auto info = trainer.train(env);
        BOOST_LOG_TRIVIAL(info)<<"Training info...";
		BOOST_LOG_TRIVIAL(info)<<info;

        // save the rewards per episode for visualization
        // purposes
        auto filename = std::string("experiments/") + EXPERIMENT_ID;
        filename += "/reinforce_rewards.csv";
        rlenvscpp::utils::::io::CSVWriter csv_writer(filename, 
		                                             rlenvscpp::utils::io::CSVWriter::default_delimiter());
        csv_writer.open();
        csv_writer.write_column_vector(trainer.episodes_total_rewards());


        // save the policy also so that we can load it and check
        // use it
        auto policy_model_filename = std::string("experiments/") + 
		                             EXPERIMENT_ID + std::string("/reinforce_cartpole_policy.pth");
        
        torch::serialize::OutputArchive archive;
        policy->save(archive);
        archive.save_to(policy_model_filename);
		
		
		BOOST_LOG_TRIVIAL(info)<<"Finished agent training";

    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    }
    catch(...){
        std::cout<<"Unknown exception occured"<<std::endl;
    }
    return 0;
}
#else
#include <iostream>
int main(){

    std::cout<<"This example requires PyTorch and gymfcpp. Reconfigure cuberl with USE_PYTORCH and USE_RLENVS_CPP flags turned ON."<<std::endl;
    return 0;
}
#endif
