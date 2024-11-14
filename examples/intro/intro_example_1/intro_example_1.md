# Example 1: _CubeRl_ Basics

In this first example we explore some of the basic elements of _cuberl_. In particular,
we will see how to read a json file, how to use boost logging, various types exposed by the library.
We will do so by using a simple Monte Carlo integration example.


## The driver code

The driver code for this tutorial is shown below. 


```cpp
/**
  * This example illustrates a simple example of Monte Carlo
  * iteration using the IterationCounter class
  *
  * */

#include "cubeai/base/cubeai_types.h"
#include "cubeai/utils/iteration_counter.h"
#include "cubeai/geom_primitives/shapes/circle.h"
#include "cubeai/extern/nlohmann/json/json.hpp"

#include <boost/log/trivial.hpp>
#include <iostream>
#include <random>
#include <fstream>

namespace intro_example_1
{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::utils::IterationCounter;
using cubeai::geom_primitives::Circle;

using json = nlohmann::json;

const std::string CONFIG = "config.json";

// read the JSON file
json
load_config(const std::string& filename){

  std::ifstream f(filename);
  json data = json::parse(f);
  return data;
}


}

int main() {

    using namespace intro_example_1;

    try{

        BOOST_LOG_TRIVIAL(info)<<"Reading configuration file...";

        // load the json configuration
        auto data = load_config(CONFIG);

        // read properties from the configuration
        const auto R = data["R"].template get<real_t>();
        const auto N_POINTS = data["N_POINTS"].template get<uint_t>();
        const auto SEED = data["SEED"].template get<uint_t>();
        const auto X = data["X"].template get<real_t>();
        const auto Y = data["Y"].template get<real_t>();

        // create a circle
        Circle c(R, {X, Y});

        // simple object to control iterations
        IterationCounter counter(N_POINTS);

        // how many points we found in the Circle
        auto points_inside_circle = 0;

        // the box has side 2
        const real_t SQUARE_SIDE = R*2.0;
        std::uniform_real_distribution dist(0.0,SQUARE_SIDE);
        std::mt19937 gen(SEED);

        BOOST_LOG_TRIVIAL(info)<<"Starting computation...";
        while(counter.continue_iterations()){
            auto x = dist(gen);
            auto y = dist(gen);
            if(c.is_inside(x,y, 1.0e-4)){
              points_inside_circle += 1;
            }
        }

        BOOST_LOG_TRIVIAL(info)<<"Finished computation...";
        auto area = (static_cast<real_t>(points_inside_circle) / static_cast<real_t>(N_POINTS)) * std::pow(SQUARE_SIDE, 2);
        BOOST_LOG_TRIVIAL(info)<<"Circle area calculated with:" <<N_POINTS<<" is: "<<area;
        BOOST_LOG_TRIVIAL(info)<<"Circle area: "<<c.area();
    }
    catch(std::exception& e){
        BOOST_LOG_TRIVIAL(error)<<e.what();
    }
    catch(...){
        BOOST_LOG_TRIVIAL(error)<<"Unknown exception occured";
    }

   return 0;
}
```

Running the code above produces the following output

```
[2024-05-30 17:51:47.253901] [0x00007fe2bbb7b540] [info]    Reading configuration file...
[2024-05-30 17:51:47.254017] [0x00007fe2bbb7b540] [info]    Starting computation...
[2024-05-30 17:51:47.258199] [0x00007fe2bbb7b540] [info]    Finished computation...
[2024-05-30 17:51:47.258222] [0x00007fe2bbb7b540] [info]    Circle area calculated with:100000 is: 3.14716
[2024-05-30 17:51:47.258226] [0x00007fe2bbb7b540] [info]    Circle area: 3.14159
```

