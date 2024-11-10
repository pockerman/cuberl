#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH


#include "cubeai/rl/algorithms/pg/simple_reinforce.h"
#include "cubeai/io/json_file_reader.h"

#include <boost/log/trivial.hpp>

namespace cubeai {
namespace rl {
namespace algos {
namespace pg {

std::ostream&
ReinforceConfig::print(std::ostream& out)const{

    out<<"Max num episodes=    "<<max_num_of_episodes<<std::endl;
    out<<"Max its per episode= "<<max_itrs_per_episode<<std::endl;
    out<<"Print frequency=     "<<print_frequency<<std::endl;
    out<<"Score queue maz size="<<scores_queue_max_size<<std::endl;
    out<<"Gamma=               "<<gamma<<std::endl;
    out<<"Exit tolerance=      "<<tolerance<<std::endl;
    out<<"Exit score level=    "<<exit_score_level<<std::endl;
    return out;
}


void 
ReinforceConfig::load_from_json(const std::string& filename){
	
	BOOST_LOG_TRIVIAL(info)<<"Loading ReinforceConfig from path: "<<filename;
	
	cubeai::io::JSONFileReader json_reader(filename);
	json_reader.open();
	
	
	normalize_rewards = json_reader.template get_value<bool>("normalize_rewards");
	max_num_of_episodes =  json_reader.template get_value<uint_t>("max_num_of_episodes");
    max_itrs_per_episode = json_reader.template get_value<uint_t>("max_itrs_per_episode");
	print_frequency = json_reader.template get_value<uint_t>("print_frequency");
    scores_queue_max_size = json_reader.template get_value<uint_t>("scores_queue_max_size");
	gamma = json_reader.template get_value<real_t>("gamma");
	tolerance = json_reader.template get_value<real_t>("tolerance");
	exit_score_level = json_reader.template get_value<real_t>("exit_score_level");
	
	BOOST_LOG_TRIVIAL(info)<<"Done loading ReinforceConfig";
	
}

}
}
}
}

#endif

