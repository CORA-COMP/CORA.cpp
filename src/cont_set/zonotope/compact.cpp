// compact - fewer generators for the same set, as CORA's zonotope.compact
//
// Syntax:   Zres = Z.compact();   Zres = Z.compact("all", tol);
// Inputs:   method - "zeros" (default) drops the generators of norm <= tol; "all" also merges
//                    those that are parallel or antiparallel up to tol; "aligned" leaves Z as it
//                    is (CORA's compact_ has no branch for it)
//           tol - CORA's default eps: relaxes "is zero" and "is parallel" (as within_tol)
// Outputs:  Zres - the zonotope without them; a batch drops what every member drops, and "all"
//           is not for a batch
// See also: reduce, box
//
// CORA: contSet/@zonotope/compact_.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The generators of norm > tol (and not all zero), CORA's nonzeroFilter; of a batch, those that
/// some member keeps.
Tensor aux_nonzero_filter(const Tensor &G, double tol) {
    const Tensor values = G.detach(); // the choice carries no gradient
    const Tensor entries = values.not_equal(values.zeros_like());
    const Tensor norm = values.vector_norm(Tensor::Norm::two, -2);
    Tensor keep = entries.any(-2).multiply(norm.greater(norm.full_like(norm.shape(), tol)));
    while (keep.ndim() > 1) keep = keep.any(0);
    return priv_select_generators(G, keep.nonzero());
}

/// Merges the generators that are parallel to a longer one (CORA's aux_deleteAligned): the
/// columns by ascending length, each kept one absorbs those parallel to it (those antiparallel
/// with a minus), by tensor operations; G is (n, m).
Tensor aux_delete_aligned(const Tensor &G_in, double tol) {
    const Tensor G0 = aux_nonzero_filter(G_in, tol);
    if (G0.shape()[0] == 1) return G0.abs_sum();
    const int64_t m = G0.shape().back();
    if (m == 0) return G0;
    // the choice is made on detached values, the merged columns carry the gradient
    const Tensor values = G0.detach();
    const Tensor length = values.vector_norm(Tensor::Norm::two, -2).reshape({m});
    const Tensor order = length.argsort();
    Tensor G = G0.take(order, -1);
    const Tensor unit = values.take(order, -1).divide(length.take(order, 0).reshape({1, m}));

    // which sorted columns are parallel (dot product 1) or antiparallel (-1) to which
    const Tensor dots = unit.matrix_transpose().matmul(unit);
    const Tensor parallel = priv_within_tol(dots, 1.0, tol);
    const Tensor anti = priv_within_tol(dots, -1.0, tol);

    Tensor keep = G.ones_like({1, m});
    for (int64_t i = 0; i < m; ++i) {
        const Tensor p = parallel.slice(0, i, i + 1), a = anti.slice(0, i, i + 1);
        const Tensor own = keep.slice(1, i, i + 1);
        const Tensor several = p.sum(1, true).greater(own.ones_like());
        const Tensor opposed = a.sum(1, true).greater(own.zeros_like());
        const Tensor merge = own.multiply(several.maximum(opposed));

        // the merged column replaces column i, the absorbed ones (not i itself) leave
        const Tensor merged = G.multiply(p).sum(-1, true) - G.multiply(a).sum(-1, true);
        const Tensor at_i = G.arange_like(m).equal(G.full_like({1}, double(i))).reshape({1, m});
        G = Tensor::where(merge.multiply(at_i), merged, G);
        const Tensor absorbed = p.multiply(at_i.negative() + 1.0).maximum(a);
        keep = Tensor::where(merge, keep.multiply(absorbed.negative() + 1.0), keep);
    }
    return priv_select_generators(G, keep.nonzero());
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::compact(const std::string &method, double tol) const {
    detail::Entry entry;
    CORA_CHECK(method == "zeros" || method == "all" || method == "aligned", Error::wrong_value,
               [&] { return "compact: the method must be 'zeros', 'all' or 'aligned', got '" +
                            method + "'"; });
    CORA_CHECK(tol >= 0, Error::wrong_value,
               [&] { return "compact: tol must be nonnegative, got " + std::to_string(tol); });
    CORA_CHECK(method != "all" || G_.ndim() == 2, Error::not_supported, [] {
        return std::string("compact: method 'all' merges generators member by member; "
                           "it supports a single zonotope, not a batch");
    });
    if (method == "aligned") return *this;
    if (method == "zeros") return {c_, aux_nonzero_filter(G_, tol)};
    return {c_, aux_delete_aligned(G_, tol)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
