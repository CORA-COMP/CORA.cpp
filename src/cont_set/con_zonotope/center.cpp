// center - a point of the constrained zonotope near its middle, as CORA's conZonotope.center
//
// Syntax:   Tensor c = cz.center();
// Inputs:   -
// Outputs:  c - the offset c without constraints; else c + G x with x the Chebyshev center of the
//           factors {|x|_inf <= 1, A x = b} (in the null space of A if the set is
//           full-dimensional); a column without entries (n, 0) for an empty set. Not a batch if
//           there are constraints
// See also: interval, is_full_dim
//
// CORA: contSet/@conZonotope/center.m

#include "cont_set/con_zonotope/private/priv.h"

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
    std::vector<double> Aineq(std::size_t(2 * H.m * H.m), 0.0);
    for (int64_t j = 0; j < H.m; ++j) {
        Aineq[j * H.m + j] = -1.0;
        Aineq[(H.m + j) * H.m + j] = 1.0;
    }
    const auto x =
        priv_chebyshev_center(Aineq, std::vector<double>(std::size_t(2 * H.m), 1.0), H.A, H.b, H.m);
    if (!x) CORA_FAIL(Error::solver_issue, "center: the linear program of the center failed");
    return aux_point(cz, *x);
}

/// The Chebyshev center in the null space of A for a full-dimensional set, mapped back.
Tensor aux_center_full(const ConZonotope &cz) {
    const Tensor &A = cz.A();
    const Tensor N = priv_null_space(A);
    // no free direction: the factors are the solution of A x = b
    if (N.shape().back() == 0) return cz.c() + cz.generators().matmul(A.lstsq(cz.b()));
    const Tensor x0 = A.pinv().matmul(cz.b());
    const int64_t k = N.shape().back();
    // |x| <= 1 for x = N y + x0 is [N; -N] y <= [1 - x0; 1 + x0]
    const Tensor rows = Tensor::concat({N, -N}, -2);
    const Tensor bounds = Tensor::concat({x0.ones_like() - x0, x0.ones_like() + x0}, -2);
    const auto y = priv_chebyshev_center(rows.data(), bounds.data(), {}, {}, k);
    if (!y) CORA_FAIL(Error::solver_issue, "center: the linear program of the center failed");
    const Tensor x = N.matmul(Tensor::like(N, *y, {k, 1})) + x0;
    return cz.c() + cz.generators().matmul(x);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor ConZonotope::center_impl() const {
    if (A_.shape()[A_.ndim() - 2] == 0) return c_;
    const HostConZonotope H = priv_host_con_zonotope(*this);
    // an empty set has no center
    if (priv_is_empty(H)) return c_.zeros_like({H.n, 0});
    return is_full_dim() ? aux_center_full(*this) : aux_center_flat(*this, H);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
