// vertices - the corners of a two-dimensional box, as CORA's interval.vertices
//
// Syntax:   V = I.vertices();
// Inputs:   -
// Outputs:  V - one polygon per batch member: lower left, lower right, upper right, upper left
// See also: project, Zonotope::vertices

#include "contSet/interval/interval.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ===========================================  MAIN  =========================================== //

std::vector<Polygon> Interval::vertices() const {
    if (dim() != 2 || inf.shape().back() != 1)
        throw std::invalid_argument("Interval::vertices: needs a two-dimensional box; project it "
                                    "first, e.g. I.project({0, 1})");
    const std::vector<double> lo = inf.data(), hi = sup.data();
    std::vector<Polygon> out;
    for (std::size_t b = 0; b + 1 < lo.size(); b += 2) // (x, y) of each batch member
        out.push_back(
            {{lo[b], lo[b + 1]}, {hi[b], lo[b + 1]}, {hi[b], hi[b + 1]}, {lo[b], hi[b + 1]}});
    return out;
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
