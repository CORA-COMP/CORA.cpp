// vertices - the corners of a two-dimensional box, as CORA's interval.vertices
//
// Syntax:   V = I.vertices();
// Inputs:   -
// Outputs:  V - one polygon per batch member: lower left, lower right, upper right, upper left
// See also: project, Zonotope::vertices
//
// CORA: contSet/@interval/vertices_.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/error.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Polygon> Interval::vertices_impl() const {
    const std::vector<double> lo = inf_.data(), hi = sup_.data();
    std::vector<Polygon> out;
    for (std::size_t b = 0; b + 1 < lo.size(); b += 2) // (x, y) of each batch member
        out.push_back(
            {{lo[b], lo[b + 1]}, {hi[b], lo[b + 1]}, {hi[b], hi[b + 1]}, {lo[b], hi[b + 1]}});
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
