/**
  * Example 0: Demonstrates the use of the DummyAlgorithm class.
  * This class exposes the basic API that most implemented RL
  * algorithms expose.
  *
  * */

#include "cubeai/base/cubeai_config.h"

#ifdef USE_RL

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dummy/dummy_algorithm.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/rl/algorithms/rl_algo_config.h"
#include "cubeai/rl/agents/dummy_agent.h"
#include "cubeai/utils/iteration_counter.h"
#include "cubeai/utils/cubeai_concepts.h"
#include "cubeai/utils/lambda_utils.h"
#include "cubeai/maths/vector_math.h"

#include <iostream>
#include <unordered_map>

namespace example_0
{

const std::string SERVER_URL = "http://0.0.0.0:8001/api";

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynMat;
using cubeai::DynVec;
using cubeai::rl::algos::DummyAlgorithm;
using cubeai::rl::algos::DummyAlgorithmConfig;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;
using cubeai::rl::agents::DummyAgent;
using cubeai::utils::IterationCounter;


}

int main() {

    using namespace example_0;

    try{

        std::vector<real_t> q{0.0, 0.0, 0.0, 0.0};
        std::vector<real_t> walk{-1.0, 1.0, 5.0};

        cubeai::maths::randomize_vec(q, walk);

        std::for_each(q.begin(),
                      q.end(),
                      cubeai::utils::cubeai_print<real_t>);

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
int main() {

	std::cout<<"This example requires to configure the library with RL support. Set USE_RL to ON and rebuild"<<std::endl;
}

#endif


