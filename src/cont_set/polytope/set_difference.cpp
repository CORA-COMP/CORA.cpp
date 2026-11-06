// set_difference - the pieces of the difference of two polytopes, as CORA's polytope.mldivide
//
// For full-dimensional sets, piece i is the part of P that lies beyond row i of S and inside the
// rows before it: {-a_i x <= -b_i, a_j x <= b_j for j < i, rows of P}. The pieces with an interior
// cover the closure of P \ S. CORA runs the same loop but keeps only the last piece (and compacts P
// twice, the second time in place of S); the lower-dimensional cases of CORA (equalities, affine
// hulls) are Error::not_supported here. The empty set minus S is empty, P minus the empty set is P.
// Not a batch.
//
// Syntax:   std::vector<Polytope> pieces = P.set_difference(S);
// Inputs:   S - a polytope of the dimension of P
// Outputs:  pieces - polytopes whose union is the closure of P \ S (none if S contains P)
// See also: Polytope::complement, Polytope::compact, Polytope::intersect
//
// CORA: contSet/@polytope/mldivide.m
// CORA authors: Adrian Kulmburg
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Polytope> Polytope::set_difference(const Polytope &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "set_difference: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    CORA_CHECK((is_h_rep() ? A_ : V_).ndim() == 2 && (S.is_h_rep() ? S.A() : S.V()).ndim() == 2,
               Error::not_supported, [] {
                   return std::string("set_difference: it supports single polytopes, not a batch");
               });
    if (priv_polytope_is_empty(*this)) return {empty(dim())};
    if (priv_polytope_is_empty(S)) return {*this};
    const Polytope P1 = compact("all", numerics::kPolytopeCompact);
    const Polytope P2 = S.compact("all", numerics::kPolytopeCompact);
    CORA_CHECK(P1.is_full_dim() && P2.is_full_dim(), Error::not_supported, [] {
        return std::string("set_difference: only full-dimensional operands are supported (CORA's "
                           "lower-dimensional branches are not ported)");
    });
    const Constraints C1 = P1.constraints(), C2 = P2.constraints();
    std::vector<Polytope> pieces;
    // piece i: beyond row i of S, inside rows 0..i-1 of S and the rows of P
    for (int64_t i = 0; i < C2.A.shape()[0]; ++i) {
        const Tensor beyond_A = -C2.A.slice(-2, i, i + 1), beyond_b = -C2.b.slice(-2, i, i + 1);
        const Polytope piece(Tensor::concat({beyond_A, C2.A.slice(-2, 0, i), C1.A}, -2),
                             Tensor::concat({beyond_b, C2.b.slice(-2, 0, i), C1.b}, -2));
        if (piece.is_full_dim()) pieces.push_back(piece);
    }
    return pieces;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
