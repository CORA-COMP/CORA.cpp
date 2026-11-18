// simplex - a simplex that contains a zonotope, as CORA's zonotope.simplex
//
// The regular simplex whose facet normals are the directions from the centroid away from its
// vertices (a Gram-Schmidt basis orthogonal to (1, ..., 1), as CORA), pushed out to the support
// function of the zonotope on every facet. CORA lists the facets as its convex hull finds them;
// here facet i is opposite vertex i. Not a batch.
//
// Syntax:   Polytope P = Z.simplex();
// Inputs:   -
// Outputs:  P - {x | A x <= b} with n + 1 halfspaces: a simplex that contains Z
// See also: Zonotope::support_function
//
// CORA: contSet/@zonotope/simplex.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "cora/cont_set/polytope/polytope.h"
#include "global/gram_schmidt.h"
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope Zonotope::simplex() const {
    detail::Entry entry;
    CORA_CHECK(detail::cora_center(*this).ndim() == 2, Error::not_supported,
               [] { return std::string("simplex: it supports a single zonotope, not a batch"); });
    const int64_t n = dim();
    // the basis orthogonal to the ones vector: column j of its remainder is vertex j of a simplex
    const Rows B = gram_schmidt(Row(std::size_t(n + 1), 1.0));
    std::vector<double> normals(std::size_t((n + 1) * n));
    for (int64_t j = 0; j <= n; ++j) {
        double norm = 0.0;
        for (int64_t k = 0; k < n; ++k)
            norm += B[std::size_t(k + 1)][std::size_t(j)] * B[std::size_t(k + 1)][std::size_t(j)];
        norm = std::sqrt(norm);
        // facet j is opposite vertex j: its outer normal points away from the vertex
        for (int64_t k = 0; k < n; ++k)
            normals[std::size_t(j * n + k)] = -B[std::size_t(k + 1)][std::size_t(j)] / norm;
    }
    const Tensor A = Tensor::like(detail::cora_center(*this), normals, {n + 1, n});
    return {A, A.matmul(detail::cora_center(*this)) +
                   A.matmul(detail::cora_generators(*this)).abs_sum()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
