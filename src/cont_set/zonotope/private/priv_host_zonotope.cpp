// priv_host_zonotope - the numbers of a zonotope on the host
//
// The host algorithms run a linear program or a search.
//
// Syntax:   HostZonotope H = priv_host_zonotope(Z);
// Inputs:   Z - a single zonotope (no batch)
// Outputs:  H - center (n) and generators (n x m row-major) as doubles, for host algorithms
// See also: priv_zonotope_norm
//
// CORA.cpp only: MATLAB hands its matrices to the solver as they are
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

HostZonotope priv_host_zonotope(const Zonotope &Z) {
    // the linear programs read CORA's n x m matrix of generator columns
    return {Z.dim(), Z.nr_generators(), Z.center().data(),
            Z.generators().matrix_transpose().data()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
