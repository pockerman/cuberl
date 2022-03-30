#ifndef TD_ALGO_BASE_H
#define TD_ALGO_BASE_H


#include "cubeai/rl/algorithms/rl_algorithm_base.h"
#include "cubeai/rl/worlds/envs_concepts.h"



//#include <deque>
//#include <vector>
//#include <iostream>

namespace cubeai {
namespace rl {
namespace algos {
namespace td {

/*struct TDAlgoConfig: RLAlgoConfig
{
    real_t gamma;
    real_t eta;
    uint_t seed{42};
};*/

///
///\brief The TDAlgoBase class. Base class
/// for deriving TD algorithms
///
template<typename EnvType>
class TDAlgoBase: public RLAlgoBase<EnvType>
{
public:

    ///
    /// \brief env_t
    ///
    typedef EnvType env_type;

    ///
    /// \brief action_t
    ///
    typedef typename env_type::action_type action_type;

    ///
    /// \brief state_t
    ///
    typedef typename env_type::state_type state_type;

    ///
    /// \brief Destructor
    ///
    virtual ~TDAlgoBase() = default;


protected:

    ///
    /// \brief DPAlgoBase
    /// \param name
    ///
    TDAlgoBase()=default;
};


}

}

}

}

#endif // TD_ALGO_BASE_H
