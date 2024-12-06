# Example 7:  Importance sampling

The vanialla Monte Carlo method . 
In this example we will look at <a href="https://astrostatistics.psu.edu/su14/lectures/cisewski_is.pdf">imporatnce sampling</a> which 
allows us to overcome the dificculty of sampling from a difficult distribution.

## Importance sampling


The first example we will consider is taken from [1]. We want to estimate the 
following probability; $P(Z > 3)$ where $Z\sim N(0,1)$ This is just the integral:

$$P(Z > 3) = \int_{3}^{+\infty}f(x)dx = \int_{-\infty}^{+\infty}h(x)f(x)dx$$

where $h(x)$ is 1 if $x > 3$ and 0 otherwise and $f(x)$ is the PDF for the standard normal distribution.

## The driver code

The driver code for this tutorial is shown below.

```cpp
#include "cubeai/base/cubeai_types.h"


#include <cmath>
#include <utility>
#include <tuple>
#include <iostream>
#include <random>
#include <algorithm>

namespace exe
{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynMat;
using cubeai::DynVec;

// create transition matrix
DynMat<real_t> create_transition_matrix(){
    return DynMat<real_t>({{0.9, 0.1}, {0.5, 0.5}});
}

// create transition matrix
DynMat<real_t>
compute_matrix_power(const DynMat<real_t>& mat, uint_t power ){

    auto result = mat;

    for(uint i=0; i<power - 1; ++i){
        result *= mat;
    }

    return result;
}

void print_matrix(const DynMat<real_t>& mat){
    std::cout<<"["<<mat(0, 0)<<" , "<<mat(0, 1)<<"]"<<std::endl;
    std::cout<<"["<<mat(1, 0)<<" , "<<mat(1, 1)<<"]"<<std::endl;
}

}

int main() {

    using namespace exe;

    auto transition = create_transition_matrix();

    std::cout<<"After 3 steps..."<<std::endl;
    // after 3 steps
    auto t_3 = compute_matrix_power(transition, 3 );
    print_matrix(t_3);

    std::cout<<"After 50 steps..."<<std::endl;
    // after 3 steps
    auto t_50 = compute_matrix_power(transition, 50 );
    print_matrix(t_50);

    std::cout<<"After 100 steps..."<<std::endl;

    // after 3 steps
    auto t_100 = compute_matrix_power(transition, 100 );
    print_matrix(t_100);

    // initial vector
    auto v1 = DynVec<real_t>(2);
    v1[0] = 1.0;
    v1[1] = 0.0;

    // We can calculate the probability of being
    // in a specific state after k iterations multiplying
    // the initial distribution and the transition matrix: v⋅Tk.

    std::cout<<"v_3="<<v1.transpose() * t_3<<std::endl;
    std::cout<<"v_50="<<v1.transpose() * t_50<<std::endl;
    std::cout<<"v_100="<<v1.transpose() * t_100<<std::endl;

    // initial vector
    v1[0] = 0.5;
    v1[1] = 0.5;

    std::cout<<"v_3="<<v1.transpose() * t_3<<std::endl;
    std::cout<<"v_50="<<v1.transpose() * t_50<<std::endl;
    std::cout<<"v_100="<<v1.transpose() * t_100<<std::endl;
    return 0;
}

```

Running the driver code above produces the following output

```bash
After 3 steps...
[0.844 , 0.156]
[0.78 , 0.22]
After 50 steps...
[0.833333 , 0.166667]
[0.833333 , 0.166667]
After 100 steps...
[0.833333 , 0.166667]
[0.833333 , 0.166667]
v_3=0.844 0.156
    0     0
v_50=0.833333 0.166667
       0        0
v_100=0.833333 0.166667
       0        0
v_3=0.422 0.078
0.422 0.078
v_50= 0.416667 0.0833333
 0.416667 0.0833333
v_100= 0.416667 0.0833333
 0.416667 0.0833333

```


## References

1. Larry Wasserman, _All of Statistics. A Concise Course in Statistical Inference_, Springer 2003.