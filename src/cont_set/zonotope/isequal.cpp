// isequal - whether a zonotope equals a zonotope, a box or a point, up to a tolerance
//
// Syntax:   res = Z.isequal(S);   res = Z.isequal(S, tol);
// Inputs:   S - a zonotope, a box or a point (n, 1);  tol - CORA's default eps: the centers agree
//           as within_tol, and so do the generators after compact "all" (up to order and sign)
// Outputs:  res - true if the two sets are the same; a different dimension gives false. Not a batch
// See also: compact, contains
//
// CORA: contSet/@zonotope/isequal.m

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/within_tol.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the columns of A and B (n x m each, row-major) agree up to order and sign: each column
/// of A takes the first unused column of B that equals it or its negative (CORA's compareMatrices).
bool aux_same_columns(const std::vector<double> &A, const std::vector<double> &B, int64_t n,
                      int64_t m, double tol) {
    std::vector<bool> used(m, false);
    const auto agree = [&](int64_t i, int64_t j, double sign) {
        for (int64_t r = 0; r < n; ++r)
            if (!within_tol(A[r * m + i], sign * B[r * m + j], tol)) return false;
        return true;
    };
    for (int64_t i = 0; i < m; ++i) {
        bool found = false;
        for (int64_t j = 0; j < m && !found; ++j)
            if (!used[j] && (agree(i, j, 1.0) || agree(i, j, -1.0))) found = used[j] = true;
        if (!found) return false;
    }
    return true;
}

/// Whether the zonotopes are equal up to tol.
bool aux_isequal(const Zonotope &Z, const Zonotope &S, double tol) {
    const HostZonotope a = priv_host_zonotope(Z.compact("all", tol)),
                       b = priv_host_zonotope(S.compact("all", tol));
    for (int64_t i = 0; i < Z.dim(); ++i)
        if (!within_tol(a.c[i], b.c[i], tol)) return false;
    return a.m == b.m && aux_same_columns(a.G, b.G, a.n, a.m, tol);
}

/// Why the call is refused, if it is: a batch (the generators are compared one set at a time), a
/// negative tol.
Refusal aux_refusal(const Tensor &c, const Tensor &c2, double tol) {
    if (c.ndim() != 2 || c2.ndim() != 2)
        return std::pair{Error::not_supported, std::string("it supports a single zonotope, not a "
                                                           "batch")};
    if (!(tol >= 0))
        return std::pair{Error::wrong_value, "tol must be nonnegative, got " + std::to_string(tol)};
    return std::nullopt;
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Zonotope::isequal(const Zonotope &S, double tol) const {
    detail::Entry entry;
    const Refusal refusal = aux_refusal(c_, S.center(), tol);
    CORA_CHECK(!refusal, refusal->first, [&] { return "isequal: " + refusal->second; });
    return S.dim() == dim() && aux_isequal(*this, S, tol);
}

bool Zonotope::isequal(const Interval &I, double tol) const {
    detail::Entry entry;
    const Refusal refusal = aux_refusal(c_, I.inf(), tol);
    CORA_CHECK(!refusal, refusal->first, [&] { return "isequal: " + refusal->second; });
    // a box is the zonotope of its axes; the empty box is no zonotope
    return I.dim() == dim() && !I.is_empty() &&
           aux_isequal(*this, priv_zonotope_of_interval(I), tol);
}

bool Zonotope::isequal(const Tensor &p, double tol) const {
    detail::Entry entry;
    CORA_CHECK(p.ndim() == 2, Error::dimension_mismatch,
               [] { return std::string("isequal: a point must be a column (n, 1)"); });
    const Refusal refusal = aux_refusal(c_, p, tol);
    CORA_CHECK(!refusal, refusal->first, [&] { return "isequal: " + refusal->second; });
    return p.shape()[0] == dim() && p.shape().back() == 1 &&
           aux_isequal(*this, priv_zonotope_of_point(p), tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
