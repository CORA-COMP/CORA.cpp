// interval - the smallest box around a polytope, as CORA's polytope.interval
//
// The empty set gives the empty box. A vertex polytope takes the minimum and maximum of its
// vertices (also for a batch), a halfspace polytope solves two linear programs per dimension (not
// a batch).
//
// Syntax:   Interval I = P.interval();
// Inputs:   P - polytope
// Outputs:  I - the least and the largest value of every coordinate, which may be infinite
// See also: box, Polytope::center
//
// CORA: contSet/@polytope/interval.m
// CORA authors: Matthias Althoff, Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Polytope::interval_impl() const {
    const int64_t n = dim();
    if (is_v_rep()) {
        if (V_.size(-2) == 0) return Interval::empty(n);
        return {V_.min(-2), V_.max(-2)};
    }
    const auto bounds = priv_box_bounds(priv_host_h(constraints()));
    if (!bounds) return Interval::empty(n);
    return {priv_vector_tensor(A_, bounds->first), priv_vector_tensor(A_, bounds->second)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
