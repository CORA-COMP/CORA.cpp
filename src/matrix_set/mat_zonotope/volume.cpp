// volume - the volume of a matrix zonotope, as CORA's matZonotope.volume
//
// The volume of the zonotope of the stacked matrices (Zonotope::volume): zero unless the generators
// span the rows * cols dimensions.
//
// Syntax:   V = M.volume();
// Inputs:   method, order - as Zonotope::volume
// Outputs:  V - the volume, shape (..., 1, 1)
// See also: zonotope, Zonotope::volume
//
// CORA: matrixSet/@matZonotope/volume.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor MatZonotope::volume(const std::string &method, int order) const {
    detail::Entry entry;
    // the nested entry does not check: the method is checked here
    const bool known = method == "exact" || method == "reduce" || method == "alamo";
    CORA_CHECK(known, Error::wrong_value, [&] {
        return "volume: the method must be 'exact', 'reduce' or 'alamo', got '" + method + "'";
    });
    return zonotope().volume(method, order);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
