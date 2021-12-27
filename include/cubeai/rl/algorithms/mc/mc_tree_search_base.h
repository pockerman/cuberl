#ifndef MC_TREE_SEARCH_BASE_H
#define MC_TREE_SEARCH_BASE_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/algorithm_base.h"
#include "cubeai/rl/algorithms/rl_algo_config.h"
#include "cubeai/utils/iteration_mixin.h"

#include "cubeai/base/cubeai_config.h"

#ifdef CUBEAI_DEBUG
#include <cassert>
#endif

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

namespace cubeai{
namespace rl{
namespace algos{


///
/// \brief The MCTreeSearchConfig struct
///
struct MCTreeSearchConfig: public RLAlgoConfig
{
    uint_t max_tree_depth = 1000;
    real_t temperature = 1.0;

};

template<typename ActionTp, typename StateTp>
class MCTreeNodeBase
{
public:

    typedef ActionTp action_type;
    typedef StateTp state_type;

    ///
    /// \brief MCTreeNodeBase
    /// \param parent
    /// \param action
    ///
    MCTreeNodeBase(std::shared_ptr<MCTreeNodeBase> parent, action_type action);

    ///
    ///
    ///
    virtual ~MCTreeNodeBase()=default;

    ///
    /// \brief add_child
    /// \param child
    ///
    void add_child(std::shared_ptr<MCTreeNodeBase<ActionTp, StateTp>> child);

    ///
    /// \brief add_child
    /// \param parent
    /// \param action
    ///
    void add_child(std::shared_ptr<MCTreeNodeBase<ActionTp, StateTp>> parent, action_type action);

    ///
    /// \brief has_children
    /// \return
    ///
    bool has_children()const noexcept{return children_.empty() != true;}

    ///
    /// \brief get_child
    /// \param cidx
    /// \return
    ///
    std::shared_ptr<MCTreeNodeBase> get_child(uint_t cidx){return children_[cidx];}

    ///
    /// \brief n_children
    /// \return
    ///
    uint_t n_children()const noexcept{return children_.size();}

    ///
    /// \brief shuffle_children
    ///
    void shuffle_children()noexcept;

    ///
    /// \brief explored_children
    /// \return
    ///
    uint_t n_explored_children()const noexcept{return explored_children_;}

    ///
    /// \brief update_visits
    ///
    void update_visits()noexcept{total_visits_ += 1;}

    ///
    ///
    ///
    void update_explored_children()noexcept{explored_children_ += 1;}

    ///
    /// \brief update_total_score
    /// \param score
    ///
    void update_total_score(real_t score)noexcept {total_score_ += score;}

    ///
    /// \brief ucb
    /// \param temperature
    /// \return
    ///
    real_t ucb(real_t temperature )const;

    ///
    /// \brief max_ucb_child
    /// \return
    ///
    std::shared_ptr<MCTreeNodeBase> max_ucb_child(real_t temperature)const;

    ///
    /// \brief win_pct
    /// \return
    ///
    real_t win_pct()const{return total_score_ / total_visits_ ;}

    ///
    /// \brief total_visits
    /// \return
    ///
    uint_t total_visits()const noexcept{return total_visits_;}

    ///
    /// \brief get_action
    /// \return
    ///
    uint_t get_action()const noexcept{return action_;}

    ///
    /// \brief parent
    /// \return
    ///
    std::shared_ptr<MCTreeNodeBase> parent(){return parent_;}

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
    /// \brief explored_children_
    ///
    uint_t explored_children_;

    ///
    /// \brief action_
    ///
    action_type action_;

    ///
    /// \brief parent_
    ///
    std::shared_ptr<MCTreeNodeBase> parent_;

    ///
    /// \brief children_
    ///
    std::vector<std::shared_ptr<MCTreeNodeBase>> children_;

};

template<typename ActionTp, typename StateTp>
MCTreeNodeBase<ActionTp, StateTp>::MCTreeNodeBase(std::shared_ptr<MCTreeNodeBase> parent, action_type action)
    :
    total_score_(0.0),
    total_visits_(0),
    explored_children_(0),
    action_(action),
    parent_(parent),
    children_()
{}


template<typename ActionTp, typename StateTp>
void
MCTreeNodeBase<ActionTp, StateTp>::add_child(std::shared_ptr<MCTreeNodeBase<ActionTp, StateTp>> child){

#ifdef CUBEAI_DEBUG
    assert(child != nullptr && "Cannot add null children");
#endif

   children_.push_back(child);

}

template<typename ActionTp, typename StateTp>
void
MCTreeNodeBase<ActionTp, StateTp>::shuffle_children()noexcept{

    if(children_.empty()){
        return;
    }

    std::random_shuffle(children_.begin(), children_.end());
}

template<typename ActionTp, typename StateTp>
void
MCTreeNodeBase<ActionTp, StateTp>::add_child(std::shared_ptr<MCTreeNodeBase<ActionTp, StateTp>> parent, action_type action){

    add_child(std::make_shared<MCTreeNodeBase<ActionTp, StateTp>>(parent, action));
}

template<typename ActionTp, typename StateTp>
std::shared_ptr<MCTreeNodeBase<ActionTp, StateTp>>
MCTreeNodeBase<ActionTp, StateTp>::max_ucb_child(real_t temperature)const{

    if(children_.empty()){
        return std::shared_ptr<MCTreeNodeBase>();
    }

    auto current_ucb = children_[0]->ucb(temperature);
    auto current_idx = 0;
    auto dummy_idx = 0;
    for(auto node : children_){

        auto node_ucb = node->ucb(temperature);

        if(node_ucb > current_ucb){

            current_idx = dummy_idx;
            current_ucb = node_ucb;
        }

        dummy_idx += 1;
    }

    return children_[current_idx];
}

template<typename ActionTp, typename StateTp>
real_t
MCTreeNodeBase<ActionTp, StateTp>::ucb(real_t temperature )const{

    return win_pct() + temperature * std::sqrt(std::log(parent_ -> total_visits()) / total_visits_);
}

///
/// \brief MCTreeSearchBase.
///
template<typename EnvTp, typename NodeTp=MCTreeNodeBase<typename EnvTp::action_type, typename EnvTp::state_type>>
class MCTreeSearchBase: public AlgorithmBase
{
public:

    static_assert (std::is_default_constructible<typename EnvTp::action_type>::value, "Action type is not default constructible");
    static_assert (std::is_default_constructible<typename EnvTp::state_type>::value, "State type is not default constructible");

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
    /// \brief simulate_node
    /// \param node
    ///
    virtual typename env_type::time_step_type simulate_node(std::shared_ptr<node_type> node, env_type& env)=0;

    ///
    /// \brief expand_node
    /// \param node
    ///
    virtual void expand_node(std::shared_ptr<node_type> node, env_type& env)=0;

    ///
    /// \brief backprop
    ///
    virtual void backprop(std::shared_ptr<node_type> node)=0;

    ///
    /// \brief max_depth_tree
    /// \return
    ///
    uint_t max_depth_tree()const noexcept{return max_depth_tree_;}

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

    ///
    /// \brief max_depth_tree_
    ///
    uint_t max_depth_tree_;

    ///
    /// \brief temperature_
    ///
    real_t temperature_;

};

template<typename EnvTp, typename NodeTp>
MCTreeSearchBase<EnvTp, NodeTp>::MCTreeSearchBase(MCTreeSearchConfig config, env_type& env)
    :
      AlgorithmBase(config.n_episodes, config.tolerance),
      itr_mix_(config.n_itrs_per_episode, config.render_episode),
      env_(env),
      root_(nullptr),
      max_depth_tree_(config.max_tree_depth),
      temperature_(config.temperature)
{}

template<typename EnvTp, typename NodeTp>
void
MCTreeSearchBase<EnvTp, NodeTp>::actions_after_training_episodes(){

    //this->AlgorithmBase::actions_after_training_episodes();
}

template<typename EnvTp, typename NodeTp>
void
MCTreeSearchBase<EnvTp, NodeTp>::actions_before_training_episodes(){
    //this->AlgorithmBase::actions_before_training_episodes();
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
