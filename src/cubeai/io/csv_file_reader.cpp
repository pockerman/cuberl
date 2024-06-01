#include "cubeai/base/cubeai_consts.h"
#include "cubeai/base/cubeai_config.h"
#include "cubeai/io/csv_file_reader.h"
#include "cubeai/io/file_formats.h"

#include <boost/algorithm/string.hpp>

#ifdef CUBEAI_DEBUG
#include <cassert>
#endif

namespace cubeai{
namespace io{

CSVFileReader::CSVFileReader(const std::string& file_name, const std::string delimeter)
    :
    FileReaderBase(file_name, FileFormats::Type::CSV),
    delimeter_(delimeter),
    current_row_idx_(0)
{}

CSVFileReader::~CSVFileReader()
{
    close();
}

/*
void
CSVFileReader::open(){

    if(!file_reader_.is_open()){

        try{
            file_reader_.open(file_name_, std::ios_base::in);

#ifdef CUBEAI_DEBUG

            if(!file_reader_.good()){
                std::string msg = "Failed to open file: " + file_name_;
                assert(false && msg.c_str());
            }
#endif

        }
        catch(...){

#ifdef CUBEAI_DEBUG
            std::string msg("Failed to open file: ");
            msg += file_name_;
            assert(false && msg.c_str());
#else
            throw;
#endif

        }
    }
}*/

/*void
CSVFileReader::close(){

    if(file_reader_.is_open()){
        file_reader_.close();
    }
}*/

std::vector<std::string>
CSVFileReader::read_line(){


    if(! this -> is_open()){
        throw std::logic_error("Attempt to read from a file that is not open");
    }

    auto& f = this->get_file_stream();

#ifdef CUBEAI_DEBUG

            if(!f.good()){
                std::string msg("Failed to open file: ");
                msg += file_name_;
                assert(false && msg.c_str());
            }
#endif
	
    std::vector<std::string> result;
    if(f.eof()){
        result.push_back(CubeAIConsts::eof_string());
        return result;
    }

    std::string line = "";
    std::getline(f, line);
    
    boost::algorithm::split(result, line, boost::is_any_of(delimeter_));
    current_row_idx_++;
    return result;
}

std::vector<uint_t>
CSVFileReader::read_line_as_uint(){

    auto line = read_line();
    std::vector<uint_t> line_int(line.size());

    for(uint_t i=0; i<line.size(); ++i){
        line_int[i] = std::stoull(line[i]);
    }

    return line_int;
}


}
}
