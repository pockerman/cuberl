#ifndef DP_ALGO_BASE_H
#define DP_ALGO_BASE_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/io/csv_file_writer.h"
#include "cubeai/rl/algorithms/rl_algorithm_base.h"

#include <tuple>
#include <vector>
#include <string>
#include <type_traits>

namespace cuberl {
namespace rl {
namespace algos {
namespace dp {

///
/// \brief The DPSolverBase class
///
template<typename EnvType>
class DPSolverBase: public RLSolverBase<EnvType>
{
public:

    ///
    /// \brief The environment type the  solver is using
    ///
    typedef typename RLSolverBase<EnvType>::env_type env_type;

    // state type should be integral
    static_assert(std::is_integral<typename EnvType::state_type>::value);
    static_assert(std::is_integral<typename EnvType::action_type>::value);

    ///
    /// \brief Destructor
    ///
    virtual ~DPSolverBase() = default;

protected:

    ///
    /// \brief DPAlgoBase
    /// \param name
    ///
    DPSolverBase()=default;


};

}
}
}
}

#endif // DP_ALGO_BASE_H
