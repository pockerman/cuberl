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
#include "cubeai/maths/matrix_utilities.h"

#include <iostream>


namespace example21{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynVec;
using cubeai::ml::classifiers::KNearestNeighbors;
using cubeai::maths::LpMetric;

class LpMetricWrapper
{
public:

    typedef real_t value_type;

    LpMetricWrapper()=default;

    template<typename DataPair>
    real_t evaluate(const DataPair& v1, const DataPair& v2)const{
        return LpMetric<2>::evaluate(v1.first, v2.first);
    }
};

}


int main(){

using namespace example21;


 try{

       auto data = cubeai::io::load_iris_data_set(false);

       std::cout<<cubeai::CubeAIConsts::info_str()<<" Number of training examples "<<std::get<0>(data).rows()<<std::endl;

       auto meta = std::get<2>(data);

       auto begin = meta.class_map.cbegin();
       auto end = meta.class_map.cend();

       while( begin != end){
           std::cout<<cubeai::CubeAIConsts::info_str()<<begin->first<<", "<<begin->second<<std::endl;
           ++begin;
       }


       KNearestNeighbors<DynVec<real_t>> classifier(std::get<0>(data).columns());

       auto comparison = [](const auto& v1, const auto& v2){
           return v1.first[0] == v2.first[0] && v1.first[1] == v2.first[1] && v1.first[2] == v2.first[2] && v1.first[3] == v2.first[3];
       };

       auto info = classifier.fit(std::get<0>(data), std::get<1>(data), comparison);
       std::cout<<cubeai::CubeAIConsts::info_str()<<info<<std::endl;

       DynVec<real_t> row = cubeai::maths::get_row(std::get<0>(data), 0);
       auto index = classifier.template predict<LpMetricWrapper>(row, 5);
       std::cout<<cubeai::CubeAIConsts::info_str()<<" class for first point "<<index<<"->"<<meta.class_map[index]<<std::endl;

       auto closest_points = classifier.template nearest_k_points<LpMetricWrapper>(row, 5);

      std::cout<<cubeai::CubeAIConsts::info_str()<<" Query point is "<<row<<std::endl;
       for(auto& p : closest_points){

           std::cout<<"Distance="<<p.first<<", "<<p.second<<std::endl;
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

