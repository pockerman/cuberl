#ifndef MAP_INPUT_RESOLVER_H
#define MAP_INPUT_RESOLVER_H

#include <any>
#include <map>
#include <stdexcept>

namespace cubeai{
namespace utils{
	
	template<typename InputT, typename OutT>
	struct InputResolver;

	template<typename OutT>
	struct InputResolver<std::map<std::string, std::any>, OutT>
	{
		typedef std::map<std::string, boost::any> input_type;
		typedef OutT out_type;

		static out_type resolve(const std::string& name, const input_type& input);

	};

	template<typename OutT>
	typename InputResolver<std::map<std::string, std::any>, OutT>::out_type
	InputResolver<std::map<std::string, std::any>, OutT>::resolve(const std::string& name,
																	const input_type& input){

		auto itr = input.find(name);

		if(itr == input.end()){
			throw std::logic_error("Property: " + name + " not in input");
		}

		return std::any_cast<OutT>(itr->second);

	}

	}
}


#endif