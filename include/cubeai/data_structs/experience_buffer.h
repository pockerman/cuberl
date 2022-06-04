#ifndef EXPERIENCE_BUFFER_H
#define EXPERIENCE_BUFFER_H

#include "cubeai/base/cubeai_types.h"

#include "boost/noncopyable.hpp"
#include "boost/circular_buffer.hpp"


#include <random>

namespace cubeai{
namespace containers {

///
/// \brief The Transition struct Default transition to use
/// with the ExperienceBuffer
///
struct Transition
{
    uint_t state;
    uint_t action;
    uint_t next_state;
    uint_t reward;

};

///
/// \brief The ExperienceBuffer class. A buffer based on
/// std::deque to accumulate items of type ExperienceTp.
///
template<typename ExperienceType>
class ExperienceBuffer: private boost::noncopyable{

public:

    typedef ExperienceType value_type ;
    typedef ExperienceType experience_type;


    //typedef typename std::deque<experience_t, allocator_t>::iterator iterator;
    //typedef typename std::deque<experience_t, allocator_t>::const_iterator const_iterator;
    //typedef typename std::deque<experience_t, allocator_t>::reverse_iterator reverse_iterator;
    //typedef typename std::deque<experience_t, allocator_t>::const_reverse_iterator const_reverse_iterator;

    ///
    /// \brief ExperienceBuffer
    ///
    explicit ExperienceBuffer(uint_t capacity);

    ///
    /// \brief append Add the experience item in the buffer
    ///
    void append(const experience_type& experience);

    ///
    /// \brief size
    /// \return
    ///
    uint_t size()const noexcept{return static_cast<uint_t>(buffer_.size());}

    ///
    /// \brief capacity
    /// \return
    ///
    uint_t capacity()const noexcept{return static_cast<uint_t>(buffer_.capacity());}

    ///
    /// \brief empty. Returns true if the buffer is empty
    /// \return
    ///
    bool empty()const noexcept{return buffer_.empty();}

    ///
    /// \brief clear
    ///
    void clear()noexcept{buffer_.clear();}

    ///
    /// \brief operator []
    /// \param idx
    /// \return
    ///
    value_type& operator[](uint_t idx){return buffer_[idx];}

    ///
    /// \brief operator []
    /// \param idx
    /// \return
    ///
    const value_type& operator[](uint_t idx)const{return buffer_[idx];}

    ///
    /// \brief sample. Sample batch_size experiences from the
    /// buffer and transfer them in the BatchTp container.
    ///
    template<typename BatchTp>
    void sample(uint_t batch_size, BatchTp& batch, uint_t seed=42)const;

    /*iterator begin(){return experience_.begin();}
    iterator end(){return experience_.end();}

    const_iterator begin()const{return experience_.begin();}
    const_iterator end()const{return experience_.end();}

    reverse_iterator rbegin(){return experience_.rbegin();}
    reverse_iterator rend(){return experience_.rend();}

    const_reverse_iterator rbegin()const{return experience_.rbegin();}
    const_reverse_iterator rend()const{return experience_.rend();}*/

private:

   ///
   /// \brief buffer_
   ///
   boost::circular_buffer<ExperienceType> buffer_;

};

template<typename ExperienceTp>
ExperienceBuffer<ExperienceTp>::ExperienceBuffer(uint_t max_size)
    :
      buffer_(max_size)
{}

template<typename ExperienceTp>
void
ExperienceBuffer<ExperienceTp>::append(const experience_type& experience){

    buffer_.push_back(experience);
}

template<typename ExperienceTp>
template<typename BatchTp>
void
ExperienceBuffer<ExperienceTp>::sample(uint_t batch_size, BatchTp& batch, uint_t seed)const{
    throw std::logic_error("This function is not implemented!!!");
}


}
}

#endif // EXPERIENCE_BUFFER_H
