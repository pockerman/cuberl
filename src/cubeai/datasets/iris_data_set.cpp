#include "cubeai/datasets/iris_data_set.h"
#include "cubeai/base/cubeai_data_paths.h"
#include "cubeai/io/csv_file_reader.h"
#include "cubeai/base/cubeai_consts.h"

#include <string>
#include <algorithm>

namespace cubeai{
namespace datasets{

std::string IrisDataSet::invalid_class_name =  "INVALID_CLASS";

IrisDataSet::IrisDataSet(bool do_load, bool append_ones)
{
    if(do_load){
        load(append_ones);
    }
}

void
IrisDataSet::random_shuffle(){
    std::random_shuffle(points_.begin(), points_.end());
}

void
IrisDataSet::load(bool append_ones){
    std::string file = CubeAIDataPaths::iris_data_path();

    io::CSVFileReader reader(file);
    reader.open();

    meta_.has_ones = append_ones;
    meta_.path = file;
    meta_.class_map.insert({0, "Iris-setosa"});
    meta_.class_map.insert({1, "Iris-versicolor"});
    meta_.class_map.insert({2, "Iris-virginica"});

    uint_t cols = append_ones? 5 : 4;
    real_t val = append_ones ? 1.0 : 0.0;

    // initialize the data
    points_.resize(150, std::pair<std::vector<real_t>, uint_t>());
    //labels_.resize(150);

    // read the first line as this is the header
    reader.read_line();

    uint_t r = 0;

     while(!reader.eof() && r < points_.size() ){

          auto line = reader.read_line();

          std::vector<real_t> row(4, 0.0);

          for(uint_t i = 0; i<line.size()-1; ++i){
             row[i] = std::atof(line[i].c_str());
          }

          auto label = CubeAIConsts::INVALID_SIZE_TYPE;
          if(line[4] == "Iris-setosa"){
              label = 0;
          }
          else if(line[4] == "Iris-versicolor"){
              label = 1;
          }
          else if(line[4] == "Iris-virginica"){
              label = 2;
          }
          else{
              throw std::invalid_argument("Unknown class in reduced iris dataset: "+line[4]);
          }

          uint c = append_ones?1:0;

          points_[r].second = label;
          points_[r].first.resize(cols, val); //assign(row.begin(), row.end());
          for(; c<cols; ++c){
              points_[r].first[c] = row[ append_ones ? c-1 : c ];
          }

        r++;
      }
}

std::string_view
IrisDataSet::get_class_name(uint_t cls)const{

    auto cls_itr = meta_.class_map.find(cls);

    if(cls_itr == meta_.class_map.end()){
        return IrisDataSet::invalid_class_name;
    }

    return cls_itr->second;
}

}
}
