// contains - whether a constrained zonotope contains a point, a zonotope or a box
//
// Syntax:   res = cz.contains(p);   res = cz.contains(S, "exact", tol);
// Inputs:   p, S, I - points (n, N), a zonotope or a box of the dimension of cz
//           method - "exact" (default) or "exact:venum"; CORA's "exact:polymax" needs Polytope,
//                    "approx", "approx:st", "sampling" and "sampling:dual" are not ported
//           tol - relaxes the norm of the points as in Zonotope::contains, and decides whether a
//                 set that is a point holds them; default: the repository tolerance
//                 (docs/architecture.md)
// Outputs:  res - true if all points are in the set. A point p is in it if (p; b) is in the
//           zonotope (c; 0) + (G; A) [-1, 1]^m, a linear program per point. A zonotope or a box
//           is in it if its corners (the sign combinations of its generators) are; of at most
//           12 generators. An empty set contains no point. Not a batch. CORA's second and third
//           outputs (the certificate and the scaling) are not returned
// See also: is_full_dim, Zonotope::contains
//
// CORA: contSet/@conZonotope/contains_.m

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr int64_t kMaxCornerGenerators = 12; // 2^12 corners, a linear program each

// Checks ------------------------------------------------------------------------------------------

/// Why the method is refused, if it is: an unknown one, or one that is not ported.
detail::Refusal aux_method_refusal(const std::string &method) {
    const bool exact = method == "exact" || method == "exact:venum";
    const bool known = exact || method == "exact:polymax" || method == "approx" ||
                       method == "approx:st" || method == "sampling" ||
                       method == "sampling:primal" || method == "sampling:dual";
    if (!known)
        return std::pair{Error::no_specific_alg, "no algorithm '" + method + "' for this set"};
    if (!exact)
        return std::pair{Error::not_supported,
                         "the method '" + method + "' is not ported; use 'exact'"};
    return std::nullopt;
}

/// Why the call is refused, if it is: a batch (a program per set), points of another dimension, or
/// a method.
detail::Refusal aux_points_refusal(const ConZonotope &cz, const Tensor &p,
                                   const std::string &method) {
    if (p.ndim() != 2 || p.shape()[0] != cz.dim())
        return std::pair{Error::dimension_mismatch,
                         "the points must be a matrix (" + std::to_string(cz.dim()) +
                             ", N) of columns, got the shape " + shape_text(p.shape())};
    if (cz.c().ndim() != 2)
        return std::pair{Error::not_supported,
                         std::string("a linear program per set: it supports a single set, "
                                     "not a batch")};
    return aux_method_refusal(method);
}

/// The same for a zonotope: also, one with more generators than the corners can be tested for.
detail::Refusal aux_zonotope_refusal(const ConZonotope &cz, const Zonotope &Z,
                                     const std::string &method) {
    if (Z.dim() != cz.dim())
        return std::pair{Error::dimension_mismatch, "the sets differ in dimension, " +
                                                        std::to_string(cz.dim()) + " and " +
                                                        std::to_string(Z.dim())};
    if (cz.c().ndim() != 2 || Z.center().ndim() != 2)
        return std::pair{Error::not_supported,
                         std::string("a linear program per set: it supports a single set, "
                                     "not a batch")};
    if (const detail::Refusal refusal = aux_method_refusal(method)) return refusal;
    if (Z.compact().generators().shape().back() > kMaxCornerGenerators)
        return std::pair{Error::not_supported,
                         std::string("a set of more than 12 generators has too many corners to "
                                     "test one by one")};
    return std::nullopt;
}

// Points ------------------------------------------------------------------------------------------

/// Whether the single constrained zonotope holds the points of P (n, N): the zonotope of the
/// centers and generators extended by the constraints holds the points extended by the offsets.
bool aux_contains_points(const ConZonotope &cz, const Tensor &P, double tol) {
    const ConZonotope compacted = priv_compact(cz);
    const int64_t N = P.shape().back(), p = compacted.A().shape()[0];
    const HostConZonotope H = priv_host_con_zonotope(compacted);
    // an empty set holds nothing but no point
    if (priv_is_empty(H)) return N == 0;
    if (N == 0) return true;
    // a set that is a point: every point lies within tol of it
    const Interval box = compacted.interval();
    if (within_tol(box.inf(), box.sup(), tol).min().item() == 1.0) {
        const Tensor point = compacted.center();
        return within_tol(P, point.broadcast_to(P.shape()), tol).min().item() == 1.0;
    }
    const Tensor extended_center =
        Tensor::concat({compacted.c(), priv_zeros(compacted.c(), p, 1)}, -2);
    const Tensor extended_generators = Tensor::concat({compacted.generators(), compacted.A()}, -2);
    const Tensor offsets = compacted.b().broadcast_to({p, N});
    return Zonotope(extended_center, extended_generators)
        .contains(Tensor::concat({P, offsets}, -2), "exact", tol);
}

// Sets --------------------------------------------------------------------------------------------

/// The corners c + G s of a zonotope for every sign vector s, as columns (n, 2^m).
Tensor aux_corners(const Zonotope &Z) {
    const int64_t m = Z.generators().shape().back(), count = int64_t{1} << m;
    std::vector<double> signs(std::size_t(m * count));
    for (int64_t j = 0; j < m; ++j)
        for (int64_t k = 0; k < count; ++k) signs[j * count + k] = ((k >> j) & 1) ? 1.0 : -1.0;
    return Z.center() + Z.generators().matmul(Tensor::like(Z.center(), signs, {m, count}));
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool ConZonotope::contains(const Tensor &p, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "contains");
    const detail::Refusal refusal = aux_points_refusal(*this, p, method);
    CORA_CHECK(!refusal, refusal->first, [&] { return "contains: " + refusal->second; });
    return aux_contains_points(*this, p, tol);
}

bool ConZonotope::contains(const Zonotope &S, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "contains");
    const detail::Refusal refusal = aux_zonotope_refusal(*this, S, method);
    CORA_CHECK(!refusal, refusal->first, [&] { return "contains: " + refusal->second; });
    // a convex set holds a zonotope exactly if it holds its corners
    return aux_contains_points(*this, aux_corners(S.compact()), tol);
}

bool ConZonotope::contains(const Interval &I, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "contains");
    CORA_CHECK(I.inf().shape().back() == 1, Error::dimension_mismatch, [&] {
        return "contains: the interval must be a box with column bounds, got the shape " +
               shape_text(I.inf().shape());
    });
    // a box is the zonotope of its axes
    const Zonotope Z = I.zonotope();
    const detail::Refusal refusal = aux_zonotope_refusal(*this, Z, method);
    CORA_CHECK(!refusal, refusal->first, [&] { return "contains: " + refusal->second; });
    return aux_contains_points(*this, aux_corners(Z.compact()), tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
