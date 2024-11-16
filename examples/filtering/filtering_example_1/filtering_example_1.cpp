/**
  * KalmanFilter example. This example is mainly taken from
  * https://github.com/AtsushiSakai/PythonRobotics/blob/master/Localization/extended_kalman_filter/extended_kalman_filter.py
  *
  * */

#include "cubeai/base/cubeai_types.h"
#include "cubeai/estimation/kalman_filter.h"
#include "cubeai/estimation/kf_model_base.h"
#include "cubeai/utils/iteration_counter.h"
#include "cubeai/io/csv_file_writer.h"

#include <boost/log/trivial.hpp>
#include <iostream>
#include <unordered_map>
#include <any>
#include <vector>

namespace example_1
{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynMat;
using cubeai::DynVec;
using cubeai::estimation::KalmanFilterConfig;
using cubeai::estimation::KalmanFilter;
using cubeai::estimation::KFMotionModelBase;
using cubeai::estimation::KFModelBase;
using cubeai::utils::IterationCounter;
using cubeai::io::CSVWriter;


real_t DT = 0.1;
real_t SIM_TIME = 50.0;

struct Cmd
{
	// v: [m/s] omega: [rad/s]
	static DynVec<real_t> cmd(real_t v = 1.0, real_t omega = 0.1);
};

DynVec<real_t> 
Cmd::cmd(real_t v, real_t omega){
	
	DynVec<real_t> u(2);
	u << v, omega;
    return u;
}

// simple struct that describes the Motion of the robots
struct MotionModel: public KFMotionModelBase<DynMat<real_t>, DynVec<real_t>>
{
	const static uint_t VEC_SIZE;
	
};

const uint_t MotionModel::VEC_SIZE = 4;

// simple struct that describes the Observation model
struct ObservationModel: public KFModelBase<DynMat<real_t>>
{
	const static uint_t VEC_SIZE;
	static DynVec<real_t> sensors();
};

const uint_t ObservationModel::VEC_SIZE = 3;

DynVec<real_t> 
ObservationModel::sensors(){ 
	DynVec<real_t> all_sensors(3);
	//all_sensors << 0.0, 0.0, 0.0;
	return all_sensors;
}


typedef MotionModel motion_model_type;
typedef ObservationModel obs_motion_type;
}

int main() {

    using namespace example_1;

    try{
		
		typedef motion_model_type::state_type state_type;
		obs_motion_type obs;
		motion_model_type motion;
		
		DynVec<real_t> x_init(4);
		x_init << 0.0, 0.0, 0.0, 0.0;
		motion.set_state(x_init);
		
		// add the matrices describing the motion
		DynMat<real_t> F(MotionModel::VEC_SIZE, MotionModel::VEC_SIZE);
		F << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
		
		motion.add_matrix("F", F);
		
		KalmanFilterConfig<motion_model_type, obs_motion_type> kf_config;
		kf_config.motion_model = &motion;
		kf_config.observation_model = &obs;
		
		// the KalmanFilter to use
		KalmanFilter<motion_model_type, obs_motion_type> kf(kf_config);
		
		// set up the matrices
		DynMat<real_t> P(MotionModel::VEC_SIZE, MotionModel::VEC_SIZE);
		P << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
		
		kf.set_matrix("P", P);
		
		DynMat<real_t> Q(MotionModel::VEC_SIZE, MotionModel::VEC_SIZE);
		
		// variance of location on x-axis 0.1
		auto x_axis_var = 0.1;
		
		// variance of location on y-axis 0.1
		auto y_axis_var = 0.1;
		
		// variance of yaw angle
		auto yaw_angle_var = 0.1;
		
		// variance of velocity
		auto v_var = 1.0;
		Q << x_axis_var, 0.0, 0.0, 0.0, 
		     0.0, y_axis_var, 0.0, 0.0,
			 0.0, 0.0, yaw_angle_var, 0.0,
			 0.0, 0.0, 0.0, v_var;
			 
		Q = Q*Q;
		kf.set_matrix("Q", Q);
		
		DynMat<real_t> R(ObservationModel::VEC_SIZE - 1, ObservationModel::VEC_SIZE -1 );
		R << 1.0, 0.0,
		     0.0, 1.0;
		kf.set_matrix("R", R);
		
		
		DynMat<real_t> B(MotionModel::VEC_SIZE, 2 );
		B << 1.0, 0.0, 0.0, 0.0,
		     0.0, 1.0, 0.0, 0.0;
		kf.set_matrix("B", B);
		
		
		// the input to the filter
		std::map<std::string, std::any > kf_input; 
		
		auto n_steps = static_cast<uint_t>(SIM_TIME / DT);
		
		BOOST_LOG_TRIVIAL(info)<<"Expected number of time steps: "<<n_steps;
		std::vector<state_type> rows;
		rows.reserve(n_steps);
		
		auto current_time = 0.0;
		while(current_time <= SIM_TIME){
			
			BOOST_LOG_TRIVIAL(info)<<"Time: "<<current_time;
			
			auto u = Cmd::cmd();
			auto w = DynVec<real_t>(u.size());
			auto z = ObservationModel::sensors();
			
			kf_input["u"] = u;
			kf_input["w"] = w;
			kf_input["z"] = z;
			
			auto& state_vec = kf.estimate(kf_input);
			
			state_type row(1 + state_vec.size());
			row[0] = current_time;
			
			for(auto i =0; i < state_vec.size(); ++i){
				row[i + 1] = state_vec[i];
			}
			
			rows.push_back(row);
			current_time += DT;
		}
		
		CSVWriter csv_writer("state.csv");
		csv_writer.open();
		
		for(auto& r:rows){
			csv_writer.write_row(r);
		}

    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    }
    catch(...){

        std::cout<<"Unknown exception occured"<<std::endl;
    }

   return 0;
}
