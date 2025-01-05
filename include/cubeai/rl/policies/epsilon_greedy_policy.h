#ifndef EPSILON_GREEDY_POLICY_H
#define EPSILON_GREEDY_POLICY_H

#include "cubeai/base/cubeai_config.h"
#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/policies/max_tabular_policy.h"
#include "cubeai/rl/policies/random_tabular_policy.h"

#ifdef USE_PYTORCH
#include "cubeai/utils/torch_adaptor.h"
#endif

#include <random>
#include <cmath>

namespace cuberl {
namespace rl {
namespace policies {


///
/// \brief The EpsilonDecayOption enum. Enumerate various decaying options.
///
enum class EpsilonDecayOption{NONE, EXPONENTIAL, INVERSE_STEP, CONSTANT_RATE};

///
/// \brief The EpsilonGreedyPolicy class
///
class EpsilonGreedyPolicy
{
public:

    constexpr static real_t MIN_EPS = 0.01;
    constexpr static real_t MAX_EPS = 1.0;
    constexpr static real_t EPSILON_DECAY_FACTOR = 0.01;

    ///
	/// \brief Constructor. Creates an epsilon-greedy tabular policy
	///
    EpsilonGreedyPolicy(real_t eps);

    ////
    ///	\brief Constructor. Creates an epsilon-greedy tabular policy
    ///
    explicit EpsilonGreedyPolicy(real_t eps, uint_t seed);


    ///
	/// \brief Constructor Creates an epsilon greedy policy with an
	///epsilon decay strategy
	///
    explicit EpsilonGreedyPolicy(real_t eps, uint_t seed, EpsilonDecayOption decay_op,
                                 real_t min_eps = MIN_EPS,
                                 real_t max_eps=MAX_EPS,
                                 real_t epsilon_decay = EPSILON_DECAY_FACTOR);

    ///
    /// \brief operator() Select action for the given state
    ///
    template<typename MapType>
    uint_t operator()(const MapType& q_map, uint_t state)const;


    /**
     * @brief Get an action i.e. index from the given values
     */
    template<typename VecType>
    uint_t operator()(const VecType& vec)const;

#ifdef USE_PYTORCH
    uint_t operator()(const torch_tensor_t& vec, torch_tensor_value_type<real_t>)const;
	uint_t operator()(const torch_tensor_t& vec, torch_tensor_value_type<float_t>)const;
	uint_t operator()(const torch_tensor_t& vec, torch_tensor_value_type<int_t>)const;
	uint_t operator()(const torch_tensor_t& vec, torch_tensor_value_type<lint_t>)const;
#endif



    /**
     * @brief any actions the policy should perform
     * on the given episode index
     */
    void on_episode(uint_t episode_idx)noexcept;

    /**
     * @brief Reset the policy
     * */
    void reset()noexcept{eps_ = eps_init_;}

    /**
     * @brief Returns the value of the epsilon
     * */
    real_t eps_value()const noexcept{return eps_;}
	
	/**
	 * @brief Set the epsilon value
	 * @param eps
	 */
	void set_eps_value(real_t eps);


    /**
     * @brief Returns the decay option
     * */
    EpsilonDecayOption decay_option()const noexcept{return decay_op_;}


private:

    real_t eps_init_;
    real_t eps_;
    real_t min_eps_;
    real_t max_eps_;
    real_t epsilon_decay_;
    EpsilonDecayOption decay_op_;

     /**
     * @brief The random engine generator
     */
    mutable std::mt19937 generator_;

    // how to select the action
    RandomTabularPolicy random_policy_;
    MaxTabularPolicy max_policy_;
};

inline
EpsilonGreedyPolicy::EpsilonGreedyPolicy(real_t eps, uint_t seed, EpsilonDecayOption decay_op,
                                         real_t min_eps, real_t max_eps, real_t epsilon_decay)
:
eps_init_(eps),
eps_(eps),
min_eps_(min_eps),
max_eps_(max_eps),
epsilon_decay_(epsilon_decay),
decay_op_(decay_op),
generator_(seed),
random_policy_(seed),
max_policy_()
{}

inline
EpsilonGreedyPolicy::EpsilonGreedyPolicy(real_t eps)
    :
      eps_init_(eps),
      eps_(eps),
      min_eps_(eps),
      max_eps_(eps),
      epsilon_decay_(eps),
      decay_op_(EpsilonDecayOption::NONE),
      random_policy_(),
      max_policy_()
{}

inline
EpsilonGreedyPolicy::EpsilonGreedyPolicy(real_t eps, uint_t seed)
    :
    EpsilonGreedyPolicy(eps, seed, EpsilonDecayOption::NONE,
                        eps, eps, eps)
{}


template<typename VecType>
uint_t
EpsilonGreedyPolicy::operator()(const VecType& vec)const{

    // generate a number in [0, 1]
    std::uniform_real_distribution<> real_dist_(0.0, 1.0);

    if(real_dist_(generator_) > eps_){
        // select greedy action with probability 1 - epsilon
        return max_policy_(vec);
    }

    // else select a random action
    return random_policy_(vec);
}

}
}
}

#endif // EPSILON_GREEDY_POLICY_H
