/**
  * Example 0: Demonstrates the use of the DummyAlgorithm class.
  * This class exposes the basic API that most implemented RL
  * algorithms expose.
  *
  * */

#include "cubeai/base/cubeai_config.h"
#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dummy/dummy_algorithm.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/rl/algorithms/rl_algo_config.h"
#include "cubeai/rl/agents/dummy_agent.h"

#include "cubeai/utils/cubeai_concepts.h"
#include "cubeai/utils/lambda_utils.h"
#include "cubeai/maths/vector_math.h"

#include "rlenvs/utils/iteration_counter.h"

#include <iostream>
#include <unordered_map>

namespace rl_example_5{

const std::string SERVER_URL = "http://0.0.0.0:8001/api";

using cuberl::real_t;
using cuberl::uint_t;
}

int main() {

    using namespace rl_example_5;

    try{

        std::vector<real_t> q{0.0, 0.0, 0.0, 0.0};
        std::vector<real_t> walk{-1.0, 1.0, 5.0};

        cuberl::maths::randomize_vec(q, walk);

        std::for_each(q.begin(),
                      q.end(),
                      cuberl::utils::cubeai_print<real_t>);

    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    }
    catch(...){

        std::cout<<"Unknown exception occured"<<std::endl;
    }

   return 0;
}
