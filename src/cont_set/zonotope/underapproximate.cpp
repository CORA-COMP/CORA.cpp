// underapproximate - vertices of a zonotope in given directions, an inner approximation
//
// The default is the generators if there are at least as many as dimensions, else
// dominant_directions (not for a batch). The vertices lie in Z, and so does their convex hull.
//
// Syntax:   V = Z.underapproximate();   V = Z.underapproximate(S);
// Inputs:   S - directions as columns (n, k); default: the generators, else dominant_directions
// Outputs:  V - (..., n, 2k): the vertices c + sum_j sign(s_i' g_j) g_j and c - the sum, per s_i
// See also: dominant_directions, vertices
//
// CORA: contSet/@zonotope/underapproximate.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::underapproximate() const {
    detail::Entry entry;
    const int64_t n = dim(), m = nr_generators();
    if (m <= n) return underapproximate(G_.matrix_transpose());
    // more generators than dimensions: the nested dominant_directions does not check
    CORA_CHECK(G_.matrix_transpose().ndim() == 2, Error::not_supported, [] {
        return std::string("underapproximate: the default directions are chosen per zonotope; it "
                           "supports a single zonotope, not a batch");
    });
    // the nonzero generators must span the space
    const int64_t nonzero = int64_t(G_.matrix_transpose()
                                        .detach()
                                        .not_equal(G_.matrix_transpose().zeros_like())
                                        .any(-2)
                                        .count_nonzero(-1)
                                        .item());
    CORA_CHECK(nonzero >= n, Error::degenerate_set, [&] {
        return "underapproximate: " + std::to_string(nonzero) +
               " nonzero generators span fewer "
               "than the " +
               std::to_string(n) + " dimensions of the zonotope";
    });
    return underapproximate(dominant_directions());
}

Tensor Zonotope::underapproximate(const Tensor &S) const {
    detail::Entry entry;
    CORA_CHECK(S.ndim() == 2 && S.shape()[0] == dim(), Error::dimension_mismatch, [&] {
        return "underapproximate: the directions must be a matrix (" + std::to_string(dim()) +
               ", k), got the shape " + shape_text(S.shape());
    });
    // sign(s_i' g_j) for every direction and generator, (..., k, m)
    const Tensor signs = S.matrix_transpose().matmul(G_.matrix_transpose()).sign();
    const Tensor offset = G_.matrix_transpose().matmul(signs.matrix_transpose()); // (..., n, k)
    const Tensor both = Tensor::stack({c_.expand_dims(-1) + offset, c_.expand_dims(-1) - offset},
                                      int(offset.ndim()));
    Shape shape = offset.shape();
    shape.back() *= 2;
    return both.reshape(shape); // columns 2i, 2i + 1: the vertex and its mirror image
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
