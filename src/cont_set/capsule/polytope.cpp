// polytope - a polytope of a capsule, as CORA's capsule.polytope
//
// Exact for the empty capsule, one dimension, a point and a segment; else "outer" is the box around
// the capsule in an orthonormal basis whose first vector is the generator's direction. The basis
// is CORA's Gram-Schmidt, the identity for g = 0 where CORA divides by 0 (cora-differences.md). The
// offsets are the support values a'c + |a'g| + r |a|, as CORALean's boundDotUp.
//
// Syntax:   P = C.polytope();   P = C.polytope(method);
// Inputs:   method - "exact" (default), "outer" or "inner"
// Outputs:  P - the polytope
// See also: zonotope, ellipsoid
//
// CORA: contSet/@capsule/polytope.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// CORA's gramSchmidt of [g, the unit vectors but the one of the largest |g_i|]: orthonormal
/// columns, the first along g; the identity for g = 0.
Tensor aux_basis(const Tensor &g, const Tensor &eye) {
    const std::vector<double> gs = g.abs().data();
    const auto pivot = std::max_element(gs.begin(), gs.end()) - gs.begin();
    if (gs[static_cast<std::size_t>(pivot)] == 0.0) return eye;
    std::vector<Tensor> columns{g};
    for (int64_t j = 0; j < eye.shape().back(); ++j)
        if (j != pivot) columns.push_back(eye.slice(-1, j, j + 1));
    std::vector<Tensor> basis;
    for (const Tensor &b : columns) {
        // v = b - sum_l (q_l'b) q_l, the sum nested to the right as CORALean's sumUp
        Tensor v = b;
        if (!basis.empty()) {
            const Tensor Q = Tensor::concat(basis, -1);
            v = b - Q.matmul(Q.matrix_transpose().matmul(b));
        }
        basis.push_back(v.divide(priv_norm(v)));
    }
    return Tensor::concat(basis, -1);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Polytope Capsule::polytope(const std::string &method) const {
    detail::Entry entry;
    CORA_CHECK(
        method == "exact" || method == "outer" || method == "inner", Error::wrong_value, [&] {
            return "polytope: the method must be 'exact', 'outer' or 'inner', got '" + method + "'";
        });
    CORA_CHECK(c_.ndim() == 2, Error::not_supported,
               [] { return std::string("polytope: it supports a single capsule, not a batch"); });
    const int64_t n = dim();
    if (is_empty()) return Polytope::empty(n);
    const double eps = machine_eps(c_.dtype());
    const auto zero = [&](const Tensor &x) { return priv_all(within_tol(x, x.zeros_like(), eps)); };
    if (n == 1) {
        // an interval: |x - c| <= |g| + r
        const Tensor reach = g_.abs().add(r_);
        return {Tensor::like(c_, {1.0, -1.0}, {2, 1}), Tensor::concat({c_ + reach, reach - c_}, 0)};
    }
    if (zero(g_) && zero(r_)) return Polytope(c_);
    // the orthonormal basis B, its first column along g
    const Tensor B = aux_basis(g_, c_.eye_like(n));
    // the negated rows as 0 - B', so that a zero stays +0 as CORALean prints it
    const Tensor Bt = B.matrix_transpose(), A = Tensor::concat({Bt, Bt.zeros_like() - Bt}, 0);
    if (zero(r_)) {
        // the segment: |b1'x| <= |g|, and b'x = 0 for the other columns, shifted by c
        const Tensor A1 = Bt.slice(0, 0, 1), Ae = Bt.slice(0, 1, n);
        const Tensor length = priv_norm(g_).reshape({1, 1});
        return {Tensor::concat({A1, A1.negative()}, 0),
                Tensor::concat({length, length}, 0) +
                    Tensor::concat({A1, A1.negative()}, 0).matmul(c_),
                Ae, Ae.matmul(c_)};
    }
    CORA_CHECK(method != "exact", Error::no_exact_alg,
               [] { return std::string("polytope: a capsule is no polytope, use 'outer'"); });
    CORA_CHECK(method != "inner", Error::not_supported,
               [] { return std::string("polytope: the method 'inner' is not supported"); });
    // the support values a'c + |a'g| + r |a| of the rows, in this order as CORALean's boundDotUp
    const Tensor length = priv_norm(A.matrix_transpose()).reshape({2 * n, 1});
    return {A, A.matmul(c_) + A.matmul(g_).abs() + r_.multiply(length)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
