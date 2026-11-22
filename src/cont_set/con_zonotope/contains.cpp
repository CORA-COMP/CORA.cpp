// contains - whether a constrained zonotope contains a point, a zonotope or a box
//
// CORA's "exact:polymax" needs Polytope; "approx", "approx:st", "sampling" and "sampling:dual"
// are not ported. tol also decides whether a set that is a point holds them. A point p is in the
// set if (p; b) is in the zonotope (c; 0) + (G; A) [-1, 1]^m, a linear program per point. A
// zonotope or a box is in it if its corners (the sign combinations of its generators) are; of at
// most 12 generators. An empty set contains no point. Not a batch. CORA's second and third
// outputs (the certificate and the scaling) are not returned.
//
// Syntax:   res = cz.contains(p);   res = cz.contains(S, "exact", tol);
// Inputs:   p, S, I - points (..., n), one per row, a zonotope or a box of the dimension of cz
//           method - "exact" (default) or "exact:venum"
//           tol - relaxes the norm of the points; default: the repository tolerance
// Outputs:  res - true if all points are in the set
// See also: is_full_dim, Zonotope::contains
// References:
//    [1] S. Sadraddini, R. Tedrake. "Linear encodings for polytope containment problems", CDC
//        2019.
//    [2] J. K. Scott, D. M. Raimondo, G. R. Marseglia, R. D. Braatz. "Constrained zonotopes: a new
//        tool for set-based estimation and fault detection", Automatica 69, 2016, pp. 126-136.
//    [3] A. Kulmburg, I. Brkan, M. Althoff. "Search-based and stochastic solutions to the zonotope
//        and ellipsotope containment problems", ECC 2024.
//
// CORA: contSet/@conZonotope/contains_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger, Matthias Althoff, Adrian Kulmburg, Ivan Brkan
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/within_tol.h"
#include "global/blocks.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

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
    if (!is_vector(p, cz.dim()))
        return std::pair{Error::dimension_mismatch,
                         "the points must be (..., " + std::to_string(cz.dim()) +
                             "), one per row, got the shape " + shape_text(p.shape())};
    if (cz.c().ndim() != 1)
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
    if (cz.c().ndim() != 1 || Z.center().ndim() != 1)
        return std::pair{Error::not_supported,
                         std::string("a linear program per set: it supports a single set, "
                                     "not a batch")};
    if (const detail::Refusal refusal = aux_method_refusal(method)) return refusal;
    if (Z.compact().nr_generators() > Zonotope::kMaxCornerGenerators)
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
        const Tensor point = compacted.center().expand_dims(-1);
        return within_tol(P, point.broadcast_to(P.shape()), tol).min().item() == 1.0;
    }
    const Tensor extended_center = Tensor::concat(
        {compacted.c().expand_dims(-1), zeros_block(compacted.c().expand_dims(-1), p, 1)}, -2);
    const Tensor extended_generators =
        Tensor::concat({compacted.generators().matrix_transpose(), compacted.A()}, -2);
    const Tensor offsets = priv_b_column(compacted).broadcast_to({p, N});
    return Zonotope(extended_center.squeeze(-1), extended_generators.matrix_transpose())
        .contains(Tensor::concat({P, offsets}, -2).matrix_transpose(), "exact", tol)
        .all();
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool ConZonotope::contains(const Tensor &p, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "contains");
    const detail::Refusal refusal = aux_points_refusal(*this, p, method);
    CORA_CHECK(!refusal, refusal->first, [&] { return "contains: " + refusal->second; });
    // the test runs on CORA's columns (n, N)
    const int64_t n = dim();
    return aux_contains_points(*this, p.reshape({p.numel() / n, n}).matrix_transpose(), tol);
}

bool ConZonotope::contains(const Zonotope &S, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "contains");
    const detail::Refusal refusal = aux_zonotope_refusal(*this, S, method);
    CORA_CHECK(!refusal, refusal->first, [&] { return "contains: " + refusal->second; });
    // a convex set holds a zonotope exactly if it holds its corners
    return aux_contains_points(*this, S.compact().corners().matrix_transpose(), tol);
}

bool ConZonotope::contains(const Interval &I, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "contains");
    CORA_CHECK(I.elem_rank() == 1, Error::dimension_mismatch, [&] {
        return "contains: the interval must be a box with column bounds, got the shape " +
               shape_text(I.inf().shape());
    });
    // a box is the zonotope of its axes
    const Zonotope Z = I.zonotope();
    const detail::Refusal refusal = aux_zonotope_refusal(*this, Z, method);
    CORA_CHECK(!refusal, refusal->first, [&] { return "contains: " + refusal->second; });
    return aux_contains_points(*this, Z.compact().corners().matrix_transpose(), tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
