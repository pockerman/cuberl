/**
  * KalmanFilter example. This example is mainly taken from
  * https://github.com/AtsushiSakai/PythonRobotics/blob/master/Localization/extended_kalman_filter/extended_kalman_filter.py
  *
  * */

#include "cubeai/base/cubeai_types.h"
#include "cubeai/estimation/kalman_filter.h"
#include "cubeai/utils/iteration_counter.h"

#include <boost/log/trivial.hpp>
#include <iostream>
#include <unordered_map>

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
};

// simple struct that describes the Observation model
struct ObservationModel: public KFModelBase<DynMat<real_t>>
{
	static DynVec<real_t> sensors();
};

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
		
		obs_motion_type obs;
		motion_model_type motion;
		
		
		// add the matrices describing the motion
		DynMat<real_t> F(4, 4);
		F << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
		
		motion.add_matrix("F", F);
		
		KalmanFilterConfig<motion_model_type, obs_motion_type> kf_config;
		kf_config.motion_model = &motion;
		kf_config.observation_model = &obs;
		
		// the KalmanFilter to use
		KalmanFilter<motion_model_type, obs_motion_type> kf(kf_config);
		
		
		std::map<std::string, DynVec<real_t> > kf_input; 
		
		auto current_time = 0.0;
		while(current_time <= SIM_TIME){
			
			BOOST_LOG_TRIVIAL(info)<<"Time: "<<current_time;
			
			auto u = Cmd::cmd();
			auto w = DynVec<real_t>(u.size());
			auto z = ObservationModel::sensors();
			
			kf_input["u"] = u;
			kf_input["w"] = w;
			kf_input["z"] = z;
			
			
			//kf.estimate(kf_input);
			
			current_time += DT;
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
