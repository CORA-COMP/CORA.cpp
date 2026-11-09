// intersect - the intersection of a constrained zonotope with a set, as CORA's and_
//
// One equality per dimension makes the points agree, c + G x = c2 + G2 x2. The result is compact,
// without zero rows and generators. A polytope adds its equalities as constraints, and each
// inequality a'x <= b the constraint a'(c + G x) + s r = m with a new factor s, where
// [m - r, m + r] = [min a'x, b] (CORA's polytope/and_; for a constrained hyperplane a halfspace
// that holds the set is skipped). The result keeps the new factors, as CORA's.
//
// Syntax:   cZres = cz.intersect(S);
// Inputs:   S - a constrained zonotope, a zonotope, a box or a polytope of the same dimension
// Outputs:  cZres - exact: the factors of both sets stay, each with its constraints
// See also: con_intersect, compact, Zonotope::intersect
//
// CORA: contSet/@conZonotope/and_.m
// CORA: contSet/@polytope/and_.m
// CORA authors: Dmitry Grebenyuk, Niklas Kochdumper, Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/tensor/tolerance.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Two constrained zonotopes -----------------------------------------------------------------------

/// Why the intersection is refused, if it is: the sets must have the same dimension.
detail::Refusal aux_refusal(const ConZonotope &cz, const ConZonotope &S) {
    if (S.dim() == cz.dim()) return std::nullopt;
    return std::pair{Error::dimension_mismatch, "intersect: the sets differ in dimension, " +
                                                    std::to_string(cz.dim()) + " and " +
                                                    std::to_string(S.dim())};
}

/// The intersection of two sets: the agreement rows of the images c + G x, without zeros.
ConZonotope aux_intersection(const ConZonotope &cz, const ConZonotope &S) {
    return priv_compact(priv_agreement(cz, S, cz.c(), cz.generators()));
}

/// The rows of R (rows, cols) as row-major host values; zero rows give none.
std::vector<double> aux_rows(const Tensor &R) {
    return R.shape()[0] == 0 ? std::vector<double>{} : R.data();
}

// A polytope --------------------------------------------------------------------------------------

/// The intersection with {x | A x <= b, Ae x = be}: the equalities as constraints, then each
/// inequality with a factor of its own (bounds as in the docstring of the file).
ConZonotope aux_intersection(const ConZonotope &cz, const Polytope &P) {
    const Polytope::Constraints H = P.constraints();
    const bool hyperplane = P.representsa("conHyperplane", numerics::kGuardHyperplane);
    const int64_t n = cz.dim();
    ConZonotope out = cz;
    if (H.Ae.shape()[0] > 0) {
        // Ae (c + G x) = be for the factors x
        const Tensor A = Tensor::concat({cz.A(), H.Ae.matmul(cz.generators())}, -2);
        const Tensor b = Tensor::concat({cz.b(), H.be - H.Ae.matmul(cz.c())}, -2);
        out = ConZonotope(cz.c(), cz.generators(), A, b);
    }
    // a constrained hyperplane takes the bounds from the set so far, else from the set after Ae
    const HostConZonotope base = priv_host_con_zonotope(out);
    const std::vector<double> rows = aux_rows(H.A), offsets = aux_rows(H.b);
    for (std::size_t i = 0; i < offsets.size(); ++i) {
        const std::vector<double> a(rows.begin() + std::ptrdiff_t(i * n),
                                    rows.begin() + std::ptrdiff_t((i + 1) * n));
        const auto range = priv_range(hyperplane ? priv_host_con_zonotope(out) : base, a);
        if (!range) return ConZonotope::empty(n); // the set is already empty
        const double lb = range->first, ub = offsets[i];
        if (lb > ub) return ConZonotope::empty(n); // beyond the halfspace (CORA would keep lb)
        if (hyperplane && range->second <= ub + numerics::kGuardContains) continue; // it holds
        // a'(c + G x) + s (lb - ub) / 2 = (ub + lb) / 2 with the new factor s in [-1, 1]
        const Tensor a_row = Tensor::like(out.c(), a, {1, n});
        const int64_t p = out.A().shape()[0];
        const Tensor slack = out.c().full_like({1, 1}, 0.5 * (lb - ub));
        const Tensor A =
            Tensor::concat({Tensor::concat({out.A(), priv_zeros(out.c(), p, 1)}, -1),
                            Tensor::concat({a_row.matmul(out.generators()), slack}, -1)},
                           -2);
        const Tensor b = Tensor::concat(
            {out.b(), out.c().full_like({1, 1}, 0.5 * (ub + lb)) - a_row.matmul(out.c())}, -2);
        const Tensor G = Tensor::concat({out.generators(), priv_zeros(out.c(), n, 1)}, -1);
        out = ConZonotope(out.c(), G, A, b);
    }
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConZonotope ConZonotope::intersect(const ConZonotope &S) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, S);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_intersection(*this, S);
}

// a zonotope and a box enter as constrained zonotopes without constraints
ConZonotope ConZonotope::intersect(const Zonotope &Z) const {
    detail::Entry entry;
    const ConZonotope S(Z);
    const detail::Refusal refusal = aux_refusal(*this, S);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_intersection(*this, S);
}

ConZonotope ConZonotope::intersect(const Interval &I) const {
    detail::Entry entry;
    const ConZonotope S(I);
    const detail::Refusal refusal = aux_refusal(*this, S);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_intersection(*this, S);
}

// a polytope: one set of its dimension
ConZonotope ConZonotope::intersect(const Polytope &P) const {
    detail::Entry entry;
    CORA_CHECK(P.dim() == dim(), Error::dimension_mismatch, [&] {
        return "intersect: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(P.dim());
    });
    CORA_CHECK(c_.ndim() == 2, Error::not_supported,
               [] { return std::string("intersect: with a polytope, a single set, not a batch"); });
    return aux_intersection(*this, P);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
