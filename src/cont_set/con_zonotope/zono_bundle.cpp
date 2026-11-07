// zono_bundle - a constrained zonotope as a zonotope bundle, as CORA's conZonotope.zonoBundle
//
// The factors are p + T y for a particular solution p and a basis T of the null space of A (as
// zonotope). Each bound |x_i| <= 1 is a slab in y; the bundle holds one zonotope per factor, the
// slab along the normalized row, with half-width sqrt(m) along the other directions of an
// orthonormal basis (Gram-Schmidt as CORA's). The bundle is mapped by G T and shifted by c + G p.
// Empty: ZonoBundle::empty. Not a batch.
//
// Syntax:   ZonoBundle bundle = cz.zono_bundle();
// Inputs:   -
// Outputs:  bundle - the same set
// See also: zonotope, polytope, ZonoBundle::con_zonotope
//
// CORA: contSet/@conZonotope/zonoBundle.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cont_set/con_zonotope/private/priv.h"
#include "cont_set/polytope/private/priv.h"
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The zonotope of the slab of the normalized row a (p entries) between -lower and upper, box of
/// half-width wide along the other basis directions, in the original coordinates.
Zonotope aux_slab(const Tensor &like, const std::vector<double> &a, double lower, double upper,
                  double wide) {
    const int64_t p = int64_t(a.size());
    std::vector<double> lo(std::size_t(p), -wide), hi(std::size_t(p), wide);
    lo[0] = -lower;
    hi[0] = upper;
    // the basis vectors are the columns of B
    const Rows basis = priv_gram_schmidt(a);
    std::vector<double> columns(static_cast<std::size_t>(p * p));
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < a.size(); ++j) columns[i * a.size() + j] = basis[j][i];
    const Tensor B = Tensor::like(like, columns, {p, p});
    return Interval(Tensor::like(like, lo, {p, 1}), Tensor::like(like, hi, {p, 1}))
        .zonotope()
        .linear_map(B);
}

} // namespace


// ===========================================  MAIN  =========================================== //

ZonoBundle ConZonotope::zono_bundle() const {
    detail::Entry entry;
    CORA_CHECK(c_.ndim() == 2, Error::not_supported,
               [] { return std::string("zono_bundle: it supports a single set, not a batch"); });
    if (A_.shape()[A_.ndim() - 2] == 0) return ZonoBundle::from(Zonotope(c_, G_));
    if (priv_is_empty(priv_host_con_zonotope(*this))) return ZonoBundle::empty(dim());
    const int64_t m = A_.shape()[1];
    const Tensor p0 = A_.pinv().matmul(b_), T = priv_null_space(A_);
    const Tensor shift = c_ + G_.matmul(p0);
    const int64_t p = T.shape()[1];
    // no free direction: the constraints fix the factors, and the set is a point
    if (p == 0) return ZonoBundle::from(Zonotope(shift, G_.zeros_like({dim(), 0})));
    // the rows [T; -T] y <= [1 - p0; 1 + p0], each normalized
    const Tensor rows = Tensor::concat({T, T * -1.0}, 0);
    const Tensor offsets = Tensor::concat({p0 * -1.0 + 1.0, p0 + 1.0}, 0);
    const Tensor length = rows.multiply(rows).sum(-1, true).sqrt();
    const std::vector<double> a = rows.divide(length).data(), u = offsets.divide(length).data();
    std::vector<Zonotope> slabs;
    slabs.reserve(static_cast<std::size_t>(m));
    for (int64_t i = 0; i < m; ++i)
        slabs.push_back(aux_slab(c_,
                                 std::vector<double>(a.begin() + i * p, a.begin() + (i + 1) * p),
                                 u[std::size_t(m + i)], u[std::size_t(i)], std::sqrt(double(m))));
    return ZonoBundle(std::move(slabs)).linear_map(G_.matmul(T)).plus(shift);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
