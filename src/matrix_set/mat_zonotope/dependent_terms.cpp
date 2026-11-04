// dependent_terms - the square and the second-order Taylor terms of a matrix zonotope, as CORA's
//
// Syntax:   auto [sq, H] = A.dependent_terms(r);
// Inputs:   r - the step;  A is square
// Outputs:  sq - r^2 M^2 for every matrix M of A, as a matrix zonotope: the center r^2 (C^2 +
//           sum G_i^2 / 2) and the generators r^2 (C G_i + G_i C), r^2 G_i^2 / 2 and r^2 (G_i G_j +
//           G_j G_i) for i < j (lexicographic), with the coefficients b_i, 2 b_i^2 - 1, b_i b_j;
//           H - I + M r + M^2 r^2 / 2: the center I + C r + sq.C / 2, the generators sq.G / 2 with
//           G_i r added to the first h
// See also: expm_mixed, IntervalMatrix::dependent_terms
//
// CORA: matrixSet/@matZonotope/dependentTerms.m

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "global/format.h"
#include "matrix_set/mat_zonotope/private/priv_stacking.h"

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

std::pair<MatZonotope, MatZonotope> MatZonotope::dependent_terms(double r) const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "dependent_terms: the matrix zonotope must be square, got the shape " +
               shape_text(C_.shape());
    });
    const int64_t h = num_generators();
    const double r_squared = r * r;
    const Tensor C = C_.expand_dims(-3);
    // the squares of the generators, whose half goes to the center
    const Tensor squares = G_.matmul(G_);
    const Tensor sq_center = C_.matmul(C_) * r_squared + (squares.sum(-3) * 0.5) * r_squared;
    Tensor sq_generators =
        priv_join((C.matmul(G_) + G_.matmul(C)) * r_squared, (squares * 0.5) * r_squared);
    if (h >= 2) sq_generators = priv_join(sq_generators, aux_pair_products(G_) * r_squared);
    const Tensor H_center = C_.eye_like(rows()) + C_ * r + sq_center * 0.5;
    // the first h generators of H also hold G_i r
    const Tensor half = sq_generators * 0.5;
    const Tensor H_generators =
        h == 0 ? half : priv_join(half.slice(-3, 0, h) + G_ * r, half.slice(-3, h, half.size(-3)));
    return {MatZonotope(sq_center, sq_generators), MatZonotope(H_center, H_generators)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
