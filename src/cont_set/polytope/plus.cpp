// plus - the Minkowski sum of a polytope with a polytope, a box or a point, as CORA's polytope.plus
//
// Exact. R^n absorbs and the empty set annihilates (up to 1e-10, CORA's tolerance), the origin is
// neutral. Two vertex sets sum to all pairs of vertices, the second varying slowest; the other
// pairs of forms sum their halfspaces (the projection of the block product). A point shifts the
// offsets b + A v and be + Ae v, and the vertices. Not a batch for halfspaces and for the checks.
//
// Syntax:   Pres = P.plus(S);   Pres = P.plus(I);   Pres = P.plus(v);
// Inputs:   S, I - a polytope or a box of the dimension of P
//           v - a column (..., n, 1)
// Outputs:  Pres - P + S, in the form of the operands (halfspaces unless both have vertices)
// See also: Polytope::minus, Polytope::linear_map
//
// CORA: contSet/@polytope/plus.m
// CORA: contSet/@polytope/private/priv_plus_minus_vector.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// All sums of a vertex of V1 (n, p1) and one of V2 (n, p2), the vertex of V2 varying slowest.
Tensor aux_vertex_sums(const Tensor &V1, const Tensor &V2) {
    std::vector<int64_t> batch = V1.shape();
    const int64_t n = batch[batch.size() - 2], p1 = batch.back(), p2 = V2.shape().back();
    batch.resize(batch.size() - 2);
    std::vector<int64_t> shape2 = batch, shape1 = batch, flat = batch;
    shape2.insert(shape2.end(), {n, p2, 1});
    shape1.insert(shape1.end(), {n, 1, p1});
    flat.insert(flat.end(), {n, p2 * p1});
    return (V2.reshape(shape2) + V1.reshape(shape1)).reshape(flat);
}

/// The sum of two halfspace polytopes: the projection of their product (R^n or empty are the
/// callers' cases).
Polytope aux_halfspace_sum(const Polytope &P, const Polytope &S) {
    const Polytope::Constraints C1 = P.constraints(), C2 = S.constraints();
    const int64_t n = P.dim();
    const std::optional<HostH> sum = priv_plus_h(priv_host_h(C1), priv_host_h(C2));
    if (!sum) return Polytope::empty(n);
    const Polytope::Constraints C = priv_tensors_h(C1.A, *sum);
    return {C.A, C.b, C.Ae, C.be};
}

/// The sum of two single polytopes, after the cases of the whole space, the empty set and the
/// origin.
Polytope aux_single_sum(const Polytope &P, const Polytope &S) {
    const double tol = numerics::kPolytopeSum;
    const int64_t n = P.dim();
    if (priv_is_fullspace(P, tol) || priv_is_fullspace(S, tol)) return Polytope::inf(n);
    if (P.representsa("emptySet", tol) || S.representsa("emptySet", tol)) return Polytope::empty(n);
    // the origin changes nothing
    if (P.representsa("origin", tol)) return S;
    if (S.representsa("origin", tol)) return P;
    if (P.is_v_rep() && S.is_v_rep()) return Polytope(aux_vertex_sums(P.V(), S.V()));
    return aux_halfspace_sum(P, S);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Sets --------------------------------------------------------------------------------------------

Polytope Polytope::plus(const Polytope &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "plus: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    const Tensor &like = is_h_rep() ? A_ : V_;
    if (like.ndim() == 2 && (S.is_h_rep() ? S.A() : S.V()).ndim() == 2)
        return aux_single_sum(*this, S);
    // a batch: all sums of the vertices
    CORA_CHECK(is_v_rep() && S.is_v_rep(), Error::not_supported, [] {
        return std::string("plus: the sum of halfspaces supports a single polytope, not a batch");
    });
    return Polytope(aux_vertex_sums(V_, S.V()));
}

Polytope Polytope::plus(const Interval &I) const { return plus(Polytope(I)); }

// A point shifts the offsets of the rows and the vertices, whatever forms are held.
Polytope Polytope::plus(const Tensor &v) const {
    detail::Entry entry;
    CORA_CHECK(is_column(v, dim()), Error::dimension_mismatch, [&] {
        return "plus: the point must be a column of " + std::to_string(dim()) +
               " entries, got the shape " + shape_text(v.shape());
    });
    Parts parts;
    if (is_h_rep()) {
        parts.A = A_;
        parts.Ae = Ae_;
        parts.b = b_ + A_.matmul(v);
        parts.be = be_ + Ae_.matmul(v);
    }
    if (is_v_rep()) parts.V = V_ + v;
    return Polytope(std::move(parts));
}

Set Polytope::translate_impl(const Tensor &v) const { return plus(v); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
