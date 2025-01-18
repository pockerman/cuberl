#include "cubeai/rl/policies/max_tabular_policy.h"
#include "cubeai/base/cubeai_config.h"
#include "rlenvs/utils/io/csv_file_writer.h"

#include <vector>
#include <algorithm>
#include <iterator>

#ifdef CUBERL_DEBUG
#include <cassert>
#endif

namespace cuberl {
namespace rl {
namespace policies {

template<>
uint_t
MaxTabularPolicy::get_action(const std::vector<std::vector<real_t>>& q_map,
                             uint_t state_idx){

#ifdef CUBERL_DEBUG
    assert(state_idx < q_map.size() && "Invalid state index. Should be state_idx < q_map.size()");
#endif

    return MaxTabularPolicy::get_action(q_map[state_idx]);
}


template<>
uint_t
MaxTabularPolicy::get_action(const DynMat<real_t>& mat,
                             uint_t state_idx){

#ifdef CUBERL_DEBUG
    assert(state_idx < mat.size() && "Invalid state index. Should be state_idx < q_map.size()");
#endif

    return MaxTabularPolicy::get_action(mat.row(state_idx));
}

void 
MaxTabularPolicy::save(const std::string& filename)const{
	
	typedef MaxTabularPolicy::state_type state_type;
	typedef MaxTabularPolicy::action_type action_type;
	
	rlenvscpp::utils::io::CSVWriter csv_writer(filename);
	csv_writer.open();
	
	csv_writer.write_column_names({"state", "action"});
	
	for(uint_t s=0; s<state_action_map_.size(); ++s){
		std::tuple<state_type, action_type> row = {s, state_action_map_[s]};
		csv_writer.write_row(row);
	}
	
	csv_writer.close();
}



void 
MaxTabularPolicyBuilder::build_from_state_action_function(const DynMat<real_t>& q,
	                                                      MaxTabularPolicy& policy){
															  
	policy.state_action_map_.clear();
	policy.state_action_map_.resize(q.rows());
	
	for(uint_t s=0; s<q.rows(); ++s){
		auto action = policy.get_action(q, s);
		policy.state_action_map_[s] = action;
	}
															  
} 

}
}
}
