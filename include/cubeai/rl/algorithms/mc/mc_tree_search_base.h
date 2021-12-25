#ifndef MC_TREE_SEARCH_BASE_H
#define MC_TREE_SEARCH_BASE_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/algorithm_base.h"
#include "cubeai/rl/algorithms/rl_algo_config.h"
#include "cubeai/utils/iteration_mixin.h"

#include <vector>
#include <memory>

namespace cubeai{
namespace rl{
namespace algos{



struct MCTreeSearchConfig: public RLAlgoConfig
{
    uint_t max_tree_depth = 1000;
    real_t temperature = 1.0;

};

template<typename ActionTp, typename StateTp>
class MCTreeNodeBase
{
public:

    virtual ~MCTreeNodeBase()=default;

    ///
    /// \brief add_child
    /// \param child
    ///
    void add_child(std::shared_ptr<MCTreeNodeBase<ActionTp, StateTp>> child);

    ///
    /// \brief update_visits
    ///
    void update_visits()noexcept{total_visits_ += 1;}

    ///
    /// \brief ucb
    /// \param temperature
    /// \return
    ///
    real_t ucb(real_t temperature )const;


protected:

    ///
    /// \brief total_score_
    ///
    real_t total_score_;

    ///
    /// \brief total_visits_
    ///
    uint_t total_visits_;

    ///
    /// \brief parent_
    ///
    std::shared_ptr<MCTreeNodeBase> parent_;

    ///
    /// \brief children_
    ///
    std::vector<std::shared_ptr<MCTreeNodeBase>> children_;

};

///
/// \brief MCTreeSearchBase.
///
template<typename EnvTp, typename NodeTp=MCTreeNodeBase<typename EnvTp::action_type, typename EnvTp::state_type>>
class MCTreeSearchBase: public AlgorithmBase
{
public:

    static_assert (std::is_default_constructible<typename EnvTp::action_type>::value, "Action type is not default constructible");
    static_assert (std::is_default_constructible<typename EnvTp::action_type>::state_type, "State type is not default constructible");

    ///
    /// \brief env_type
    ///
    typedef EnvTp env_type;

    ///
    /// \brief node_type
    ///
    typedef NodeTp node_type;

    ///
    /// \brief actions_before_training_episodes. Execute any actions the
    /// algorithm needs before starting the iterations
    ///
    virtual void actions_before_training_episodes() override;

    ///
    /// \brief actions_after_training_episodes. Actions to execute after
    /// the training iterations have finisehd
    ///
    virtual void actions_after_training_episodes() override;

    ///
    /// \brief step Do one step of the algorithm
    ///
    virtual void on_episode() = 0;

    ///
    /// \brief expand_node
    /// \param node
    ///
    virtual void expand_node(std::shared_ptr<node_type> node)=0;

    ///
    /// \brief backprop
    ///
    virtual void backprop(std::shared_ptr<node_type> node);


protected:

    ///
    /// \brief MCTreeSearchBase
    /// \param config
    ///
    MCTreeSearchBase(MCTreeSearchConfig config, env_type& env);

    ///
    /// \brief itr_mix_
    ///
    IterationMixin itr_mix_;

    ///
    /// \brief env_
    ///
    env_type& env_;

    ///
    /// \brief root_
    ///
    std::shared_ptr<node_type> root_;

};

template<typename EnvTp, typename NodeTp>
MCTreeSearchBase<EnvTp, NodeTp>::MCTreeSearchBase(MCTreeSearchConfig config, env_type& env)
    :
      AlgorithmBase(config.n_episodes, config.tolerance),
      itr_mix_(config.n_itrs_per_episode, config.render_episode),
      env_(env)
{}


template<typename EnvTp, typename NodeTp>
void
MCTreeSearchBase<EnvTp, NodeTp>::actions_before_training_episodes(){
    this->AlgorithmBase::actions_before_training_episodes();
}

template<typename EnvTp, typename NodeTp>
void
MCTreeSearchBase<EnvTp, NodeTp>::on_episode(){

    while(itr_mix_.continue_iterations()){


    }
}


}
}
}

#endif // MC_TREE_SEARCH_BASE_H
