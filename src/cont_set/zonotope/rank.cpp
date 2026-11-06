// rank - the rank of the generators of a zonotope
//
// Syntax:   r = Z.rank();   r = Z.rank(tol);
// Inputs:   tol - singular values up to tol count as zero; default: the repository tolerance
//                 (docs/architecture.md), where CORA's own default is 0
// Outputs:  r - the number of singular values of the generators above tol; 0 without generators
// See also: is_full_dim, compact
//
// CORA: contSet/@zonotope/rank.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

int64_t Zonotope::rank(Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, G_.dtype(), "rank");
    CORA_CHECK(G_.ndim() == 2, Error::not_supported, [] {
        return std::string("rank: a rank per zonotope: it supports a single zonotope, not a batch");
    });
    return G_.shape().back() == 0 ? 0 : G_.matrix_rank(tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
