#include "cubeai/geom_primitives/geometry_utils.h"

namespace cubeai{
namespace geom_primitives{

const GeomPoint<3>
cross_product(const GeomPoint<3>& o1, const GeomPoint<3>& o2){return GeomPoint<3>();}

const GeomPoint<3>
cross_product(const GeomPoint<2>& o1, const GeomPoint<2>& o2){return GeomPoint<3>();}


}
}
