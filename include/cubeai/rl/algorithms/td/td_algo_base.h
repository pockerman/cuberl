#ifndef TD_ALGO_BASE_H
#define TD_ALGO_BASE_H

#include "cubeai/rl/algorithms/algorithm_base.h"
#include "cubeai/rl/algorithms/rl_algorithm_base.h"
#include "cubeai/rl/algorithms/rl_algo_config.h"
#include "cubeai/rl/worlds/envs_concepts.h"
#include "cubeai/rl/worlds/discrete_world.h"
#include "cubeai/rl/worlds/envs_concepts.h"
#include "cubeai/io/csv_file_writer.h"

#include <unordered_map>
#include <deque>
#include <vector>
#include <iostream>

namespace cubeai {
namespace rl {
namespace algos {
namespace td {

struct TDAlgoConfig: RLAlgoConfig
{
    real_t gamma;
    real_t eta;
    uint_t seed{42};
};

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
    typedef EnvTp env_type;


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

    ///
    /// \brief actions_before_training_episodes. Execute any actions the
    /// algorithm needs before starting the iterations
    ///
    virtual void actions_before_training_episodes();

    ///
    /// \brief actions_after_training_episodes. Actions to execute after
    /// the training iterations have finisehd
    ///
    virtual void actions_after_training_episodes();

    ///
    /// \brief gamma
    ///
    real_t gamma()const noexcept{return gamma_;}

    ///
    /// \brief eta
    /// \return
    ///
    real_t eta()const noexcept{return eta_;}

    ///
    /// \brief seed
    /// \return
    ///
    uint_t seed()const noexcept{return seed_;}






    ///
    /// \brief reset
    ///
    virtual void reset();

    ///
    /// \brief save. Save the value function
    /// to a CSV file. Applications can override this
    /// behaviour if not suitable.
    /// \param filename
    ///
    virtual void save(const std::string& filename)const;

    ///
    /// \brief save_avg_scores
    /// \param filename
    ///
    virtual void save_avg_scores(const std::string& filename)const;

    ///
    /// \brief save_avg_scores
    /// \param filename
    ///
    virtual void save_state_action_function(const std::string& filename)const;

    ///
    /// \brief make_value_function
    ///
    virtual void make_value_function();


protected:

    ///
    /// \brief DPAlgoBase
    /// \param name
    ///
    TDAlgoBase(TDAlgoConfig config, env_type& env);

     ///
     /// \brief DPAlgoBase
     /// \param name
     ///
     TDAlgoBase(uint_t n_episodes, real_t tolerance, real_t gamma,
                real_t eta, uint_t plot_f, uint_t max_num_iterations_per_episode, env_type& env);


private:



     ///
     /// \brief seed_
     ///
     uint_t seed_;

};

template<envs::discrete_world_concept EnvTp>
TDAlgoBase<EnvTp>::TDAlgoBase(uint_t n_episodes, real_t tolerance, real_t gamma,
                              real_t eta, uint_t render_env_frequency,
                              uint_t n_itrs_per_episode, env_type& env)
    :
    AlgorithmBase(n_episodes, tolerance),
    gamma_(gamma),
    eta_(eta),
    seed_(42),
    env_(env),
    v_(),
    q_(),
    rewards_(),
    iterations_()

{
    this->n_itrs_per_episode_ = n_itrs_per_episode;
    this->render_env_frequency_ = render_env_frequency;
}

template<envs::discrete_world_concept EnvTp>
TDAlgoBase<EnvTp>::TDAlgoBase(TDAlgoConfig config, env_type& env)
    :
    AlgorithmBase(config),
    gamma_(config.gamma),
    eta_(config.eta),
    seed_(config.seed),
    env_(env),
    v_(),
    q_()
{}

template<envs::discrete_world_concept EnvTp>
void
TDAlgoBase<EnvTp>::reset(){

    this->AlgorithmBase::reset();

    env_ref_().reset();
    q_.clear();

    for(uint_t s=0; s<env_ref_().n_states(); ++s){
        q_.insert_or_assign(s, DynVec<real_t>(env_ref_().n_actions(), 0.0));
    }

    tmp_scores_.clear();
    tmp_scores_.resize(this->render_env_frequency_);
    avg_scores_.clear();
    avg_scores_.resize(this->n_episodes());
}

template<envs::discrete_world_concept EnvTp>
void
TDAlgoBase<EnvTp>::actions_before_training_episodes(){
    reset();
    rewards_.resize(this->n_episodes());
    iterations_.resize(this->n_episodes());
}


template<envs::discrete_world_concept EnvTp>
void
TDAlgoBase<EnvTp>::actions_after_training_episodes(){
    make_value_function();
}

template<envs::discrete_world_concept EnvTp>
void
TDAlgoBase<EnvTp>::make_value_function(){

    // get the number of states
    auto n_states = env_ref_().n_states();
    v_.resize(n_states, 0.0);

    //not all states may have been visited
    auto itr_b = q_.begin();
    auto itr_e = q_.end();

    for(; itr_b != itr_e; ++itr_b){
        v_[itr_b->first] = blaze::max(itr_b->second);
    }
}

template<envs::discrete_world_concept EnvTp>
void
TDAlgoBase<EnvTp>::save(const std::string& filename)const{

    if(v_.size() == 0){
        return;
    }

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

template<envs::discrete_world_concept EnvTp>
void
TDAlgoBase<EnvTp>::save_avg_scores(const std::string& filename)const{

    CSVWriter writer(filename, ',', true);

    std::vector<std::string> columns(2);
    columns[0] = "Episode Id";
    columns[1] = "Value";
    writer.write_column_names(columns);

    for(uint_t s=0; s < avg_scores().size(); ++s){
        auto row = std::make_tuple(s, avg_scores()[s]);
        writer.write_row(row);
    }
}

/*template<envs::discrete_world_concept EnvTp>
void
TDAlgoBase<EnvTp>::save_state_action_function(const std::string& filename)const{

    CSVWriter writer(filename, ',', true);

    std::vector<std::string> columns(1 + env_ref_().n_actions());
    columns[0] = "State Id";

    for(uint_t a=0; a<env_ref_().n_actions(); ++a){
        columns[a + 1] = "Action-" + std::to_string(a);
    }

    writer.write_column_names(columns);

    //not all states may have been visited
    auto itr_b = q_.begin();
    auto itr_e = q_.end();

    for(; itr_b != itr_e;  ++itr_b){

        auto vals = itr_b->second;
        auto row = std::make_tuple(itr_b->first, vals[0], vals[1], vals[2], vals[3]);
        writer.write_row(row);
    }
}*/


}

}

}

}

#endif // TD_ALGO_BASE_H
