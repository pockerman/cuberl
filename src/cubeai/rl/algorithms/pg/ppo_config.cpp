#include "cubeai/rl/algorithms/pg/ppo_config.h"
#include "rlenvs/utils/io/json_file_reader.h"
#include <boost/log/trivial.hpp>

namespace cuberl {
namespace rl {
namespace algos {
namespace pg {
	
std::ostream&
PPOConfig::print(std::ostream& out)const{

    out<<"Max its per episode= "<<max_itrs_per_episode<<std::endl;
    //out<<"Normalize rewards  = "<<normalize_rewards<<std::endl;
    out<<"Gamma              = "<<gamma<<std::endl;
    return out;
}


void 
PPOConfig::load_from_json(const std::string& filename){
	
	BOOST_LOG_TRIVIAL(info)<<"Loading PPOConfig from path: "<<filename;
	
	rlenvscpp::utils::io::JSONFileReader json_reader(filename);
	json_reader.open();
	
	//normalize_rewards = json_reader.template get_value<bool>("normalize_rewards");
    max_itrs_per_episode = json_reader.template get_value<uint_t>("max_itrs_per_episode");
	gamma = json_reader.template get_value<real_t>("gamma");
	BOOST_LOG_TRIVIAL(info)<<"Done loading PPOConfig";
	
}

}
}
}
}