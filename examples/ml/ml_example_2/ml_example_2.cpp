/**
  * Example 22:
  *
  */

#include "cubeai/base/cubeai_config.h"
#include "cubeai/base/cubeai_types.h"
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
       KNearestNeighbors<DynVec<real_t>, LpMetric<2>> classifier;

       classifier.fit(data.first, data.second);


}
catch(std::exception& e){
   std::cout<<e.what()<<std::endl;
}
catch(...){

   std::cout<<"Unknown exception occured"<<std::endl;
}

return 0;
}

