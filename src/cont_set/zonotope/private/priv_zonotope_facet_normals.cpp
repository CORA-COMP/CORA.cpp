// priv_zonotope_facet_normals - the unit normals of the facets of a full-dimensional zonotope
//
// The cross products of the (n-1)-subsets of generators, normalized, without those that vanish
// (dependent generators): the first half of the rows of CORA's polytope(Z).A, in CORA's order (the
// subsets in lexicographic order; CORALean's hPolytopeFacets lists them in the reverse). The
// arithmetic is CORALean's: each minor by expansion along its first row with the sums nested to the
// right (memoized over the columns left, so the cost is 2^(n-1) rather than (n-1)!), the length by
// the sum of squares and the square root, the normal by division. Its value is exactly that; with
// gradients tracked it carries the gradient of the projection of the cross product onto the
// orthogonal complement of the generators, which has the same value up to rounding.
//
// Syntax:   Tensor C = priv_zonotope_facet_normals(Z);
// Inputs:   Z - a single zonotope without zero generators whose generators span the space
// Outputs:  C - the normals as rows (k, n)
// See also: Zonotope::minnorm, Zonotope::mink_diff, Zonotope::polytope
//
// CORA: contSet/@zonotope/polytope.m
// CORA authors: Niklas Kochdumper, Matthias Althoff
// Authors:       Tobias Ladner, Claude Sonnet 5.5
// Written:       03-November-2026
// Last update:   12-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"

#include <algorithm>
#include <cmath>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Determinants ------------------------------------------------------------------------------------

/// CORALean's detCandidate of the k x k row-major matrix A: the expansion along the first row left,
/// the terms (-1)^p a_p minor_p summed as t_0 + (t_1 + (... + (t_(d-1) + 0))). A minor is the
/// determinant of the later rows over the columns left, so it is kept per set of columns.
double aux_determinant(const std::vector<double> &A, int64_t k) {
    std::vector<double> det(std::size_t(1) << k);
    det[0] = 1.0;
    for (std::size_t mask = 1; mask < det.size(); ++mask) {
        std::vector<int64_t> columns;
        for (int64_t j = 0; j < k; ++j)
            if (mask >> j & 1) columns.push_back(j);
        const int64_t left = int64_t(columns.size()), row = k - left;
        double sum = 0.0;
        for (int64_t p = left - 1; p >= 0; --p) {
            const double a = A[row * k + columns[p]];
            sum = (p % 2 == 0 ? a : -a) * det[mask ^ (std::size_t(1) << columns[p])] + sum;
        }
        det[mask] = sum;
    }
    return det.back();
}

/// CORALean's facetNormal of the n x (n-1) matrix Q (row-major): entry i is (-1)^i times the
/// determinant of Q without row i.
std::vector<double> aux_cross(const std::vector<double> &Q, int64_t n) {
    const int64_t k = n - 1;
    std::vector<double> v(n), minor(k * k);
    for (int64_t i = 0; i < n; ++i) {
        int64_t row = 0;
        for (int64_t r = 0; r < n; ++r) {
            if (r == i) continue;
            for (int64_t j = 0; j < k; ++j) minor[row * k + j] = Q[r * k + j];
            ++row;
        }
        const double det = aux_determinant(minor, k);
        v[i] = (i % 2 == 0) ? det : -det;
    }
    return v;
}

// Lengths -----------------------------------------------------------------------------------------

/// The vector divided by its length: the squares summed as CORALean's sqSumUp does, nested to the
/// right.
void aux_normalize(std::vector<double> &v) {
    double sum = 0.0;
    for (std::size_t i = v.size(); i-- > 0;) sum = std::abs(v[i]) * std::abs(v[i]) + sum;
    const double length = std::sqrt(sum);
    for (double &x : v) x /= length;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor priv_zonotope_facet_normals(const Zonotope &Z) {
    const HostZonotope H = priv_host_zonotope(Z);
    const Tensor &G = Z.generators();
    if (H.n == 1) return Tensor::like(G, {1.0}, {1, 1});
    const int64_t k = H.n - 1;
    // the subsets of k generators in lexicographic order, CORA's
    std::vector<int64_t> subsets, subset(k);
    for (int64_t i = 0; i < k; ++i) subset[i] = i;
    while (H.m >= k) {
        subsets.insert(subsets.end(), subset.begin(), subset.end());
        int64_t i = k - 1;
        while (i >= 0 && subset[i] == H.m - k + i) --i;
        if (i < 0) break;
        ++subset[i];
        for (int64_t j = i + 1; j < k; ++j) subset[j] = subset[j - 1] + 1;
    }
    // the subsets whose cross product is not zero, with the normalized product on the host
    std::vector<int64_t> columns;
    std::vector<double> unit, cross;
    for (int64_t s = 0; s < int64_t(subsets.size()) / std::max<int64_t>(k, 1); ++s) {
        std::vector<double> Q(H.n * k);
        for (int64_t r = 0; r < H.n; ++r)
            for (int64_t j = 0; j < k; ++j) Q[r * k + j] = H.G[r * H.m + subsets[s * k + j]];
        std::vector<double> v = aux_cross(Q, H.n);
        if (std::all_of(v.begin(), v.end(), [](double x) { return x == 0.0; })) continue;
        columns.insert(columns.end(), subsets.begin() + s * k, subsets.begin() + (s + 1) * k);
        cross.insert(cross.end(), v.begin(), v.end());
        aux_normalize(v);
        unit.insert(unit.end(), v.begin(), v.end());
    }
    const int64_t rows = int64_t(columns.size()) / k;
    if (rows == 0) return G.zeros_like({0, H.n});
    const Tensor value = Tensor::like(G, unit, {rows, H.n});
    if (!G.requires_grad()) return value;
    // the gradient: w - Q (Q'Q)^-1 Q' w, normalized, is the normal up to rounding; only it enters
    const Tensor Q = G.take(columns, -1).reshape({H.n, rows, k}).permute_dims({1, 0, 2});
    const Tensor w = Tensor::like(G, cross, {rows, H.n, 1});
    const Tensor Qt = Q.matrix_transpose();
    const Tensor orthogonal = w - Q.matmul(Qt.matmul(Q).solve(Qt.matmul(w)));
    const Tensor projected =
        orthogonal.divide(orthogonal.vector_norm(Tensor::Norm::two, -2, true)).reshape({rows, H.n});
    return value + (projected - projected.detach());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
