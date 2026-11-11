// project - the projection of a polynomial zonotope onto some dimensions
//
// Exact: the rows of c, G and GI are picked, the factors stay.
//
// Syntax:   S = pZ.project(dims);
// Inputs:   dims - the 0-based dimensions to keep, in this order
// Outputs:  S - the projection
// See also: linear_map
//
// CORA: contSet/@polyZonotope/project.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"
#include "global/select.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set PolyZonotope::project_impl(const std::vector<int64_t> &dims) const {
    // the rows of the identity that are kept, so it runs on every backend and has gradients
    const Tensor P = select_dims(c_, dims, dim());
    return PolyZonotope(P.matmul(c_), P.matmul(G_), P.matmul(GI_), E_, id_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
