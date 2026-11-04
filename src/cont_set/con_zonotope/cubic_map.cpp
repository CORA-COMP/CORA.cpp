// cubic_map - an enclosure of a cubic map of constrained zonotopes, as CORA's conZonotope.cubMap
//
// Syntax:   cZres = cz.cubic_map(T);   cZres = cz.cubic_map(T, ind);
//           cZres = cz.cubic_map(cz2, cz3, T);
// Inputs:   T - cubic map: T[i][k] is an (n, n) matrix, output i is z_i = sum_k x_k (x' T[i][k] x)
//           ind - for every output i the k that enter it (CORA's default: all columns of T)
//           cz2, cz3 - the sets of the second and third factor of (x1' T x2) x3
// Outputs:  cZres - the generators of the zonotope map (the monomials of the factors, each a new
//           free factor, the linear ones first); the constraints of every set stay on the factors
//           of its own linear monomials. All of them are kept, which CORA does not for a factor
//           that the map does not depend on (the image could then miss points,
//           docs/cora-differences.md). CORA's first step, tightening the factors (rescale), is not
//           ported yet; it only makes the result smaller
// See also: Zonotope::cubic_map, quad_map
//
// CORA: contSet/@conZonotope/cubMap.m

#include "cont_set/con_zonotope/private/priv.h"
#include "cont_set/zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The constraint matrix A (..., p, m) of a set in the factors of the result: column j of A goes
/// to the column positions[j] of the (..., p, total) matrix, the others are zero.
Tensor aux_scatter(const Tensor &A, const std::vector<int64_t> &positions, int64_t total) {
    const int64_t m = A.shape().back();
    std::vector<double> selection(std::size_t(m * total), 0.0);
    for (int64_t j = 0; j < m; ++j) selection[j * total + positions[j]] = 1.0;
    return A.matmul(Tensor::like(A, selection, {m, total}));
}

/// The columns of the generators of a product of the sets of the sizes m[0], m[1], m[2] that
/// hold the linear monomial of factor j of set s: the monomials are ordered by the third set's
/// index, then the second's, then the first's, the center first.
std::vector<int64_t> aux_linear_positions(const std::array<int64_t, 3> &m, int s) {
    const int64_t N1 = m[0] + 1, Nq = N1 * (m[1] + 1);
    const int64_t stride = s == 0 ? 1 : s == 1 ? N1 : Nq;
    std::vector<int64_t> positions;
    // monomial index (factor + 1) * stride, and the center is not a generator
    for (int64_t j = 0; j < m[s]; ++j) positions.push_back((j + 1) * stride - 1);
    return positions;
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConZonotope ConZonotope::cubic_map(const std::vector<std::vector<Tensor>> &T,
                                   const std::vector<std::vector<int64_t>> &ind) const {
    detail::Entry entry;
    const std::string problem = priv_cubic_map_problem(T, ind, dim());
    CORA_CHECK(problem.empty(), Error::wrong_value, [&] { return "cubic_map: " + problem; });
    const Zonotope Z = Zonotope(c_, G_).cubic_map(T, ind);
    const int64_t m = G_.shape().back(), total = Z.generators().shape().back();
    // the monomials of the factors alone are the first generators
    std::vector<int64_t> linear;
    for (int64_t j = 0; j < m; ++j) linear.push_back(j);
    return {Z.center(), Z.generators(), aux_scatter(A_, linear, total), b_};
}

ConZonotope ConZonotope::cubic_map(const ConZonotope &cz2, const ConZonotope &cz3,
                                   const std::vector<std::vector<Tensor>> &T,
                                   const std::vector<std::vector<int64_t>> &ind) const {
    detail::Entry entry;
    const std::string problem = priv_cubic_map_problem(T, ind, dim());
    CORA_CHECK(problem.empty(), Error::wrong_value, [&] { return "cubic_map: " + problem; });
    CORA_CHECK(cz2.dim() == dim() && cz3.dim() == dim(), Error::dimension_mismatch, [&] {
        return "cubic_map: the sets differ in dimension, " + std::to_string(dim()) + ", " +
               std::to_string(cz2.dim()) + " and " + std::to_string(cz3.dim());
    });
    const Zonotope Z = Zonotope(c_, G_).cubic_map(Zonotope(cz2.c(), cz2.generators()),
                                                  Zonotope(cz3.c(), cz3.generators()), T, ind);
    const int64_t total = Z.generators().shape().back();
    const std::array<const ConZonotope *, 3> sets{this, &cz2, &cz3};
    const std::array<int64_t, 3> sizes{G_.shape().back(), cz2.generators().shape().back(),
                                       cz3.generators().shape().back()};
    // the rows of each set, on its own factors
    std::vector<Tensor> A, b;
    for (int s = 0; s < 3; ++s) {
        A.push_back(aux_scatter(sets[s]->A(), aux_linear_positions(sizes, s), total));
        b.push_back(sets[s]->b());
    }
    return {Z.center(), Z.generators(), Tensor::concat(A, -2), Tensor::concat(b, -2)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
