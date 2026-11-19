// dependent_terms - the square and the second-order Taylor terms of a matrix zonotope, as CORA's
//
// sq has the center r^2 (C^2 + sum G_i^2 / 2) and the generators r^2 (C G_i + G_i C), r^2 G_i^2 /
// 2 and r^2 (G_i G_j + G_j G_i) for i < j (lexicographic), with the coefficients b_i, 2 b_i^2 -
// 1, b_i b_j. The generators of H are sq.G / 2 with G_i r added to the first h.
//
// Syntax:   auto [sq, H] = mat_exp::dependent_terms(A, r);
// Inputs:   r - the step;  A is square
// Outputs:  sq - r^2 M^2 for every matrix M of A, as a matrix zonotope
//           H - I + M r + M^2 r^2 / 2: the center I + C r + sq.C / 2
// See also: expm_mixed, mat_exp::dependent_terms
//
// CORA: matrixSet/@matZonotope/dependentTerms.m
// CORA authors: Matthias Althoff, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/private/priv_stacking.h"
#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// G_i G_j + G_j G_i for every pair i < j of the generators (..., h, n, n), in lexicographic
/// order of the pairs.
Tensor aux_pair_products(const Tensor &G) {
    std::vector<int64_t> first, second;
    for (int64_t i = 0; i < G.size(-3); ++i)
        for (int64_t j = i + 1; j < G.size(-3); ++j) {
            first.push_back(i);
            second.push_back(j);
        }
    const Tensor Gi = G.take(first, -3), Gj = G.take(second, -3);
    return Gi.matmul(Gj) + Gj.matmul(Gi);
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::pair<Zonotope, Zonotope> mat_exp::dependent_terms(const Zonotope &A, double r) {
    detail::Entry entry;
    CORA_CHECK(A.center().size(-2) == A.center().size(-1), Error::dimension_mismatch, [&] {
        return "dependent_terms: the matrix zonotope must be square, got the shape " +
               shape_text(A.center().shape());
    });
    const int64_t h = A.nr_generators();
    const double r_squared = r * r;
    const Tensor C = A.center().expand_dims(-3);
    // the squares of the generators, whose half goes to the center
    const Tensor squares = A.generators().matmul(A.generators());
    const Tensor sq_center =
        A.center().matmul(A.center()) * r_squared + (squares.sum(-3) * 0.5) * r_squared;
    Tensor sq_generators =
        priv_join((C.matmul(A.generators()) + A.generators().matmul(C)) * r_squared,
                  (squares * 0.5) * r_squared);
    if (h >= 2)
        sq_generators = priv_join(sq_generators, aux_pair_products(A.generators()) * r_squared);
    const Tensor H_center =
        A.center().eye_like(A.center().size(-2)) + A.center() * r + sq_center * 0.5;
    // the first h generators of H also hold G_i r
    const Tensor half = sq_generators * 0.5;
    const Tensor H_generators = h == 0 ? half
                                       : priv_join(half.slice(-3, 0, h) + A.generators() * r,
                                                   half.slice(-3, h, half.size(-3)));
    return {Zonotope::matrix(sq_center, sq_generators), Zonotope::matrix(H_center, H_generators)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
