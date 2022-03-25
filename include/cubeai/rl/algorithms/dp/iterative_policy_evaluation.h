#ifndef ITERATIVE_POLICY_EVALUATION_H
#define ITERATIVE_POLICY_EVALUATION_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dp/dp_algo_base.h"
#include "cubeai/rl/policies/discrete_policy_base.h"
#include "cubeai/io/csv_file_writer.h"
#include "cubeai/utils/iteration_counter.h"
#include "cubeai/io/csv_file_writer.h"

#include <chrono>
#include <cmath>
#include <memory>

namespace cubeai{
namespace rl{

namespace algos {
namespace dp {


struct IterativePolicyEvalConfig
{
    real_t gamma{1.0};
    real_t tolerance{1.0e-6};
    std::string save_path{""};
};

///
/// \brief The IterativePolicyEval class
///
template<typename EnvType, typename PolicyType>
class IterativePolicyEval final: public DPAlgoBase<EnvType>
{
public:

    ///
    /// \brief env_type
    ///
    typedef typename DPAlgoBase<EnvType>::env_type env_type;

    ///
    /// \brief policy_type
    ///
    typedef PolicyType policy_type;

    ///
    /// \brief IterativePolicyEval
    ///
    explicit IterativePolicyEval(IterativePolicyEvalConfig config, policy_type& policy);

    ///
    /// \brief actions_before_training_begins. Execute any actions the
    /// algorithm needs before starting the iterations
    ///
    virtual void actions_before_training_begins(env_type& env)override;

    ///
    /// \brief actions_after_training_ends. Actions to execute after
    /// the training iterations have finisehd
    ///
    virtual void actions_after_training_ends(env_type& env)override;

    ///
    /// \brief actions_before_training_episode
    ///
    virtual void actions_before_episode_begins(env_type&, uint_t /*episode_idx*/)override{}

    ///
    /// \brief actions_after_training_episode
    ///
    virtual void actions_after_episode_ends(env_type&, uint_t /*episode_idx*/)override{}

    ///
    /// \brief on_episode Do one on_episode of the algorithm
    ///
    virtual EpisodeInfo on_training_episode(env_type& env, uint_t episode_idx) override;

    ///
    ///
    ///
    void save(const std::string& filename)const;

protected:


    IterativePolicyEvalConfig config_;
    ///
    /// \brief v_
    ///
    DynVec<real_t> v_;

    ///
    /// \brief policy_
    ///
    policy_type& policy_;

};

template<typename EnvType, typename PolicyType>
IterativePolicyEval<EnvType, PolicyType>::IterativePolicyEval(IterativePolicyEvalConfig config, policy_type& policy)
    :
      DPAlgoBase<EnvType>(),
      config_(config),
      v_(),
      policy_(policy)
{}

template<typename EnvType, typename PolicyType>
void
IterativePolicyEval<EnvType, PolicyType>::actions_before_training_begins(env_type& env){
    v_ = blaze::generate(env.n_states(), []( size_t /*index*/ ){ return 0.0; } );
}

template<typename EnvType, typename PolicyType>
void
IterativePolicyEval<EnvType, PolicyType>::actions_after_training_ends(env_type& /*env*/){

    if(config_.save_path != ""){
        CSVWriter file_writer(config_.save_path, CSVWriter::default_delimiter(), true);
        file_writer.write_column_names({"state_index", "value_function"});
        std::vector<real_t> row(1);
        for(auto item: v_){
            row[0] = item;
            file_writer.write_row(row);
        }
    }
}

template<typename EnvType, typename PolicyType>
EpisodeInfo
IterativePolicyEval<EnvType, PolicyType>::on_training_episode(env_type& env, uint_t episode_idx){

    auto start = std::chrono::steady_clock::now();
    auto episode_rewards = 0.0;
    auto delta = 0.0;


    cubeai::utils::IterationCounter itr_counter(env.n_states());
    uint_t s = 0;
    while(itr_counter.continue_iterations()){
    //for(uint_t s=0; s < env.n_states(); ++s){

        // every time we query itr_counter we increase the
        // counter so we miss the zero state
        auto old_v = v_[s];

        auto new_v = 0.0;

        auto state_actions_probs = policy_(s);

        for(const auto& action_prob : state_actions_probs){

            auto aidx = action_prob.first;
            auto action_p = action_prob.second;

            // get transition dynamic from the environment
            auto transition_dyn = env.p(s, aidx);

            for(auto& dyn: transition_dyn){
                auto prob = std::get<0>(dyn);
                auto next_state = std::get<1>(dyn);
                auto reward = std::get<2>(dyn);
                new_v += action_p * prob * (reward + config_.gamma * v_[next_state]);
                episode_rewards += reward;
            }
        }

        delta = std::max(delta, std::fabs(old_v - new_v));
        v_[s] = new_v;
        s += 1;
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<real_t> elapsed_seconds = end-start;

    EpisodeInfo info;
    info.episode_index = episode_idx;
    info.episode_reward = episode_rewards;
    info.episode_iterations = itr_counter.current_iteration_index();
    info.total_time = elapsed_seconds;

    if( delta < config_.tolerance){
        info.stop_training = true;
    }

    return info;    
}

template<typename EnvType, typename PolicyType>
void
IterativePolicyEval<EnvType, PolicyType>::save(const std::string& filename)const{

    CSVWriter writer(filename, ',', true);

    std::vector<std::string> columns(2);
    columns[0] = "State Id";
    columns[1] = "Value";
    writer.write_column_names(columns);

    for(uint_t s=0; s < v_.size(); ++s){
        auto row = std::make_tuple(s, v_[s]);
        writer.write_row(row);
    }
}


}

}
}
}

#endif // ITERATIVE_POLICY_EVALUATION_H
