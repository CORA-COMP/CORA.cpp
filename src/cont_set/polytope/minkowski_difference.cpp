// minkowski_difference - the Minkowski difference of a polytope and a set, as CORA's minkDiff
//
// {x | x + S in P}. "exact" (default) lowers the offset of every row by the support function of S
// along it (the set is empty if that is Inf) and an equality's offset by the least value of S
// along it, which must equal the largest. "exact:vertices" intersects the translates P - v of a
// vertex v of S (a polytope) or, for a zonotope or a box, shifts by -center and intersects the
// pairs P + g and P - g of its generators (CORA's code for these two reads undefined variables).
// R^n stays R^n. A point v gives P - v. Not a batch.
//
// Syntax:   Pres = P.minkowski_difference(S);   Pres = P.minkowski_difference(S, "exact:vertices");
// Inputs:   S - a set with a support function, or a point (n, 1)
//           type - "exact" (default) or "exact:vertices"
// Outputs:  Pres - the difference, in halfspace form
// See also: Polytope::plus, Polytope::minus, Polytope::intersect
//
// CORA: contSet/@polytope/minkDiff.m
// CORA: contSet/@polytope/private/priv_minkDiff.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// By the support function -------------------------------------------------------------------------

/// The difference by the support function of S along the rows (CORA's priv_minkDiff).
Polytope aux_exact(const Polytope &P, const ContSet &S) {
    const Polytope::Constraints C = P.constraints();
    HostH H = priv_host_h(C);
    const int64_t n = P.dim();
    for (std::size_t i = 0; i < H.A.size(); ++i) {
        const double upper = S.support_function(priv_column_tensor(C.A, H.A[i])).item();
        if (std::isinf(upper)) return Polytope::empty(n);
        H.b[i] -= upper;
    }
    for (std::size_t i = 0; i < H.Ae.size(); ++i) {
        // an equality needs S flat along it: the least and the largest value agree
        Row negated = H.Ae[i];
        for (double &x : negated) x = -x;
        const double upper = S.support_function(priv_column_tensor(C.A, H.Ae[i])).item();
        const double lower = -S.support_function(priv_column_tensor(C.A, negated)).item();
        if (!within_tol(lower, upper)) return Polytope::empty(n);
        H.be[i] -= lower;
    }
    const Polytope::Constraints R = priv_tensors_h(C.A, H);
    return {R.A, R.b, R.Ae, R.be};
}

// By shifts ---------------------------------------------------------------------------------------

/// The intersection of the translates P - v of the vertices v (columns of V).
Polytope aux_vertex_translates(const Polytope &P, const Tensor &V) {
    Polytope result = P.minus(V.slice(-1, 0, 1));
    for (int64_t k = 1; k < V.shape().back(); ++k)
        result = result.intersect(P.minus(V.slice(-1, k, k + 1)));
    return result;
}

/// The difference by shifts: for a polytope the translates of its vertices; for a zonotope or a box
/// the center, then the pair P + g and P - g of every generator.
Polytope aux_by_vertices(const Polytope &P, const ContSet &S) {
    switch (S.class_id()) {
    case ClassId::polytope:
        // the translates of the vertices of S
        return aux_vertex_translates(
            P, priv_points_tensor(P.is_h_rep() ? P.A() : P.V(), P.dim(),
                                  priv_polytope_vertices(static_cast<const Polytope &>(S))));
    case ClassId::interval:
    case ClassId::zonotope: {
        const Zonotope Z = S.class_id() == ClassId::interval
                               ? static_cast<const Interval &>(S).zonotope()
                               : static_cast<const Zonotope &>(S);
        Polytope result = P.minus(Z.center());
        for (int64_t k = 0; k < Z.generators().shape().back(); ++k) {
            const Tensor g = Z.generators().slice(-1, k, k + 1);
            result = result.plus(g).intersect(result.minus(g));
        }
        return result;
    }
    default:
        CORA_FAIL(Error::noops,
                  "minkowski_difference: no 'exact:vertices' for a " + S.class_name());
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

Polytope Polytope::minkowski_difference(const Tensor &p) const { return plus(-p); }

Set Polytope::minkowski_difference_impl(const ContSet &S2,
                                        const std::optional<std::string> &type_arg) const {
    const std::string type = type_arg.value_or("exact");
    // the generic entry checks only the dimension: the type and the batch are this class's
    if (type != "exact" && type != "exact:vertices")
        CORA_FAIL(Error::wrong_value,
                  "minkowski_difference: the type must be exact or exact:vertices, got '" + type +
                      "'");
    if ((is_h_rep() ? A_ : V_).ndim() != 2)
        CORA_FAIL(Error::not_supported,
                  "minkowski_difference: it supports a single polytope, not a batch");
    if (priv_is_fullspace(*this)) return inf(dim());
    return type == "exact" ? aux_exact(*this, S2) : aux_by_vertices(*this, S2);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
