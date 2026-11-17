// center - a point of the constrained zonotope near its middle, as CORA's conZonotope.center
//
// The Chebyshev center is taken over the factors {|x|_inf <= 1, A x = b} (in the null space of A
// if the set is full-dimensional); an empty set gives a column without entries (n, 0). Not a
// batch if there are constraints.
//
// Syntax:   Tensor c = cz.center();
// Inputs:   -
// Outputs:  c - the offset c without constraints; else c + G x with x the Chebyshev center
// See also: interval, is_full_dim
//
// CORA: contSet/@conZonotope/center.m
// CORA authors: Niklas Kochdumper, Adrian Kulmburg
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "global/blocks.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The point c + G x of the factors x, which are host numbers.
Tensor aux_point(const ConZonotope &cz, const std::vector<double> &x) {
    const Tensor factors = Tensor::like(cz.c(), x, {int64_t(x.size()), 1});
    return cz.c() + cz.generators().matmul(factors);
}

/// The Chebyshev center of the factors in a set with a flat image: the box and the equalities.
Tensor aux_center_flat(const ConZonotope &cz, const HostConZonotope &H) {
    const auto x = H.factors.chebyshev_center();
    if (!x) CORA_FAIL(Error::solver_issue, "center: the linear program of the center failed");
    return aux_point(cz, *x);
}

/// The Chebyshev center in the null space of A for a full-dimensional set, mapped back.
Tensor aux_center_full(const ConZonotope &cz) {
    const Tensor &A = cz.A();
    const Tensor N = null_space(A);
    // no free direction: the factors are the solution of A x = b
    if (N.shape().back() == 0) return cz.c() + cz.generators().matmul(A.lstsq(cz.b()));
    const Tensor x0 = A.pinv().matmul(cz.b());
    const int64_t k = N.shape().back();
    // |x| <= 1 for x = N y + x0 is [N; -N] y <= [1 - x0; 1 + x0], a polytope of the y
    const Polytope Y(Tensor::concat({N, -N}, -2),
                     Tensor::concat({x0.ones_like() - x0, x0.ones_like() + x0}, -2));
    constexpr double kInf = std::numeric_limits<double>::infinity();
    const auto y = Y.linear_programs(std::vector<double>(std::size_t(k), -kInf),
                                     std::vector<double>(std::size_t(k), kInf))
                       .chebyshev_center();
    if (!y) CORA_FAIL(Error::solver_issue, "center: the linear program of the center failed");
    const Tensor x = N.matmul(Tensor::like(N, *y, {k, 1})) + x0;
    return cz.c() + cz.generators().matmul(x);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor ConZonotope::center_impl() const {
    if (A().shape()[A().ndim() - 2] == 0) return c_;
    const HostConZonotope H = priv_host_con_zonotope(*this);
    // an empty set has no center
    if (priv_is_empty(H)) return c_.zeros_like({H.n, 0});
    return is_full_dim() ? aux_center_full(*this) : aux_center_flat(*this, H);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
