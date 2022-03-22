/**
  * Example 22:
  *
  */

#include "cubeai/base/cubeai_config.h"
#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/cubeai_consts.h"
#include "cubeai/io/data_set_loaders.h"
#include "cubeai/ml/classifiers/k_nearest_neighbors.h"
#include "cubeai/maths/lp_metric.h"

#include <iostream>


namespace example21{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynVec;
using cubeai::ml::classifiers::KNearestNeighbors;
using cubeai::maths::LpMetric;

}


int main(){

using namespace example21;


 try{

       auto data = cubeai::io::load_iris_data_set(false);

       std::cout<<cubeai::CubeAIConsts::info_str()<<" Number of training examples "<<data.first.rows()<<std::endl;

       KNearestNeighbors<DynVec<real_t>> classifier(data.first.columns());

       auto comparison = [](const auto& v1, const auto& v2){
           return v1.first[0] == v2.first[0] && v1.first[1] == v2.first[1] && v1.first[2] == v2.first[2] && v1.first[3] == v2.first[3];
       };

       classifier.fit(data.first, data.second, comparison);

}
catch(std::exception& e){
   std::cout<<e.what()<<std::endl;
}
catch(...){

   std::cout<<"Unknown exception occured"<<std::endl;
}

return 0;
}

