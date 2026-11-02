// vertices_3d - the eight corners of a three-dimensional box, as CORA's interval.vertices
//
// Syntax:   V = I.vertices_3d();
// Inputs:   -
// Outputs:  V - one list of eight points per batch member
// See also: vertices, Zonotope::vertices_3d
//
// CORA: contSet/@interval/vertices_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/error.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<std::vector<Point3>> Interval::vertices_3d_impl() const {
    const std::vector<double> lo = inf_.data(), hi = sup_.data();
    std::vector<std::vector<Point3>> out;
    for (std::size_t b = 0; b + 2 < lo.size(); b += 3) { // (x, y, z) of each batch member
        std::vector<Point3> corners;
        for (int k = 0; k < 8; ++k) // bit i of k picks the upper bound of coordinate i
            corners.push_back(
                {(k & 1 ? hi : lo)[b], (k & 2 ? hi : lo)[b + 1], (k & 4 ? hi : lo)[b + 2]});
        out.push_back(corners);
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
