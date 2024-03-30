/**
  * This example illustrates a simple example of Monte Carlo
  * iteration using the IterationCounter class
  *
  * */

#include "cubeai/base/cubeai_config.h"
#include "cubeai/base/cubeai_types.h"
#include "cubeai/utils/iteration_counter.h"
#include "cubeai/utils/cubeai_concepts.h"
#include "cubeai/geom_primitives/shapes/circle.h"

#include <iostream>
#include <random>

namespace example
{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynMat;
using cubeai::DynVec;
using cubeai::utils::IterationCounter;
using cubeai::geom_primitives::Circle;

const real_t R = 1.0;
const real_t X = 1.0;
const real_t Y = 1.0;
const uint_t N_POINTS = 100000;
const uint_t SEED = 42;
const real_t SQUARE_SIDE = R*2.0;

}

int main() {

    using namespace example;

    try{

        Circle c(R, {X, Y});

        IterationCounter counter(N_POINTS);

        auto points_inside_circle = 0;

        // the box has side 2
        std::uniform_real_distribution dist(0.0,SQUARE_SIDE);
        std::mt19937 gen(SEED);
        while(counter.continue_iterations()){
            std::cout<<"Iteration index: "<<counter.current_iteration_index()<<std::endl;
            auto x = dist(gen);
            auto y = dist(gen);

            if(c.is_inside(x,y, 1.0e-4)){
              points_inside_circle += 1;
            }
        }

        auto area = (static_cast<real_t>(points_inside_circle) / static_cast<real_t>(N_POINTS)) * std::pow(SQUARE_SIDE, 2);
        std::cout<<"Circle area calculated with:" <<N_POINTS<<" is: "<<area<<std::endl;
        std::cout<<"Circle area: "<<c.area()<<std::endl;

    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    }
    catch(...){

        std::cout<<"Unknown exception occured"<<std::endl;
    }

   return 0;
}


