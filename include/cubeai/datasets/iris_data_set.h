#ifndef IRIS_DATA_SET_H
#define IRIS_DATA_SET_H

#include "cubeai/base/cubeai_types.h"
#include <vector>
#include <utility>
#include <map>
#include <string>
#include <ostream>
#include <algorithm>

namespace cubeai {
namespace datasets {

class IrisDataSet
{
public:

    ///
    /// \brief n_rows
    /// \return
    ///
    static uint_t n_rows()noexcept{return 150;}

    ///
    /// \brief n_columns
    /// \return
    ///
    static uint_t n_columns()noexcept{return 4;}

    ///
    /// \brief INVALID_CLASS
    ///
    static std::string invalid_class_name;

    typedef std::vector<real_t>::value_type value_type;
    typedef std::vector<real_t> point_type;
    typedef std::vector<std::pair<std::vector<real_t>, uint_t>>::iterator iterator;
    typedef std::vector<std::pair<std::vector<real_t>, uint_t>>::const_iterator const_iterator;

    struct IrisMeta
    {
        std::string path;
        bool has_ones;
        std::map<uint_t, std::string> class_map;
    };


    ///
    /// \brief Constructor
    ///
    IrisDataSet(bool do_load=true, bool append_ones=false);

    ///
    /// \brief operator []
    /// \param idx
    /// \return
    ///
    std::pair<std::vector<real_t>, uint_t> operator[](uint_t idx)const{return point(idx);}

    ///
    /// \brief get_meta
    /// \return
    ///
    const IrisMeta& get_meta()const{return meta_;}

    ///
    /// \brief point
    /// \param idx
    /// \return
    ///
    std::pair<std::vector<real_t>, uint_t> point(uint_t idx)const;

    ///
    /// \brief get_class_name
    /// \return
    ///
    std::string_view get_class_name(uint_t cls)const;

    ///
    /// \brief random_shuffle
    ///
    void random_shuffle();

    ///
    /// \brief load
    ///
    void load(bool append_ones=false);

    iterator begin(){return points_.begin();}
    iterator end(){return points_.end();}

    const_iterator begin()const{return points_.begin();}
    const_iterator end()const{return points_.end();}

    ///
    /// \brief copy_data
    /// \return
    ///
    std::vector<std::pair<std::vector<real_t>, uint_t>> copy_data()const{return points_;}

private:

    IrisMeta meta_;
    std::vector<std::pair<std::vector<real_t>, uint_t>> points_;

};


inline
std::pair<std::vector<real_t>, uint_t>
IrisDataSet::point(uint_t idx)const{
    return points_[idx];
}

inline
std::ostream& operator<<(std::ostream& out, const IrisDataSet::IrisMeta& meta){
    out<<"Path="<<meta.path<<std::endl;
    out<<"Has ones column="<<std::boolalpha<<meta.has_ones<<std::endl;
    out<<"Class map..."<<std::endl;
    std::for_each(meta.class_map.begin(), meta.class_map.end(),
                  [&out](const auto& pair){
        out<<"\t"<<pair.first<<"->"<<pair.second<<std::endl;
    });

    return out;
}

inline
std::ostream& operator<<(std::ostream& out, const IrisDataSet& dataset){
    out<<"Number of rows="<<dataset.n_rows()<<std::endl;
    out<<"Number of columns"<<dataset.n_columns()<<std::endl;
    out<<dataset.get_meta()<<std::endl;
    return out;
}



}

}

#endif // IRIS_DATA_SET_H
