#ifndef REINFORCE_CONFIG_H
#define REINFORCE_CONFIG_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/utils/train_enum_type.h"

namespace cuberl {
namespace rl {
namespace algos {
namespace pg {
	

	

///
/// \struct ReinforceConfig configuration class for REINFORCE algorithm
/// \brief The ReinforceOpts struct. Holds various
/// configuration options for the Reinforce algorithm
///
struct ReinforceConfig
{
	bool normalize_rewards{false};
    
	///
	/// \brief How to train the algorithm
	///
	cuberl::utils::TrainEnumType train_type{cuberl::utils::TrainEnumType::BATCH};
 
	///
	/// \brief Max number of iterations per
	/// episode
	///
	uint_t max_itrs_per_episode;
	
	
	uint_t n_episodes;
	
	///
	/// \brief The discount factor
	///
    real_t gamma;
    
	///
	/// \brief The device type that PyTorch calculations take place
	///
	DeviceType device_type;

    ///
    /// \brief print
    /// \param out
    /// \return
    ///
    std::ostream& print(std::ostream& out)const;
	
	///
	/// \brief Load the configuration from the given json file
	///
	void load_from_json(const std::string& filename);
};


inline
std::ostream& operator<<(std::ostream& out, ReinforceConfig opts){
    return opts.print(out);
}

}
}
}
}

#endif