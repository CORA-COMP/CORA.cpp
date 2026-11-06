// con_zonotope - the constrained zonotope of a polytope, as CORA's polytope.conZonotope
//
// Syntax:   ConZonotope cz = P.con_zonotope();   cz = P.con_zonotope("exact:vertices");
//           cz = P.con_zonotope("exact:supportFunc", box);
// Inputs:   P - bounded polytope;  method - "exact:supportFunc" (default) takes the box around
//           the set (the argument, else interval()) as the zonotope and the rows as constraints
//           with slack factors; "exact:vertices" does so around the box of the vertices, the
//           slack from their extremes (Theorem 1 of Scott et al., Automatica 2016)
// Outputs:  cz - the same set as a constrained zonotope; the empty set gives
//           ConZonotope::empty(n); R^n and any unbounded set are Error::special_error. Not a
//           batch
// See also: Polytope::interval, ConZonotope::ConZonotope
//
// CORA: contSet/@polytope/conZonotope.m
// CORA: contSet/@polytope/private/priv_conZonotope_supportFunc.m
// CORA: contSet/@polytope/private/priv_conZonotope_vertices.m
//
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The error of a set that has no constrained zonotope.
[[noreturn]] void aux_unbounded() {
    CORA_FAIL(Error::special_error, "Polytope is unbounded and can therefore not be converted "
                                    "into a constrained zonotope.");
}

/// The constrained zonotope around the box [lo, hi] (columns) of the rows A_all x <= or = b_all
/// whose bound of the row i over the set is sigma_i: the factors of the box and one slack factor
/// per row, (sigma - b)/2 wide.
ConZonotope aux_around_box(const Tensor &lo, const Tensor &hi, const Tensor &A_all,
                           const Tensor &b_all, const Tensor &sigma) {
    const int64_t n = lo.shape()[0], rows = A_all.shape()[0];
    const Tensor c = (lo + hi) * 0.5, G = ((hi - lo) * 0.5).diag();
    return {c, Tensor::concat({G, G.zeros_like({n, rows})}, -1),
            Tensor::concat({A_all.matmul(G), ((sigma - b_all) * 0.5).diag()}, -1),
            (b_all + sigma) * 0.5 - A_all.matmul(c)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConZonotope Polytope::con_zonotope(const std::string &method,
                                   const std::optional<Interval> &box) const {
    detail::Entry entry;
    CORA_CHECK(method == "exact:vertices" || method == "exact:supportFunc", Error::wrong_value,
               [&] {
                   return "con_zonotope: the method must be 'exact:vertices' or "
                          "'exact:supportFunc', got '" +
                          method + "'";
               });
    const int64_t n = dim();
    CORA_CHECK(!box || box->dim() == n, Error::dimension_mismatch, [&] {
        return "con_zonotope: the box must have the dimension of the polytope, " +
               std::to_string(n);
    });
    if (priv_is_fullspace(*this)) aux_unbounded();
    if (priv_has_no_vertices(*this)) return ConZonotope::empty(n);
    const Tensor &like = is_h_rep() ? A_ : V_;
    if (method == "exact:vertices") {
        if (!priv_polytope_is_bounded(*this)) aux_unbounded();
        const Rows vertices = priv_polytope_vertices(*this);
        if (vertices.empty()) return ConZonotope::empty(n);
        const Constraints C = constraints();
        const Tensor V = priv_points_tensor(like, n, vertices);
        const Tensor A_all = Tensor::concat({C.A, C.Ae}, 0), b_all = Tensor::concat({C.b, C.be}, 0);
        // the least value of each row over the vertices
        return aux_around_box(V.min(-1, true), V.max(-1, true), A_all, b_all,
                              A_all.matmul(V).min(-1, true));
    }
    const Interval B = box ? *box : interval_impl();
    if (B.is_empty()) return ConZonotope::empty(n);
    if (B.inf().detach().isinf().max().item() > 0.0 || B.sup().detach().isinf().max().item() > 0.0)
        aux_unbounded();
    const Constraints C = constraints();
    const Tensor A_all = Tensor::concat({C.A, C.Ae}, 0), b_all = Tensor::concat({C.b, C.be}, 0);
    // the least value of an inequality's row over the box: the lower bound where the entry is
    // positive, the upper where it is negative; an equality's offset is its own
    const Tensor lo = B.inf().matrix_transpose(), hi = B.sup().matrix_transpose();
    const Tensor corner = Tensor::where(C.A.less(C.A.zeros_like()), hi, lo);
    const Tensor sigma_inequalities =
        C.A.shape()[0] == 0 ? C.b : C.A.multiply(corner).sum(-1, true);
    const Tensor sigma = Tensor::concat({sigma_inequalities, C.be}, 0);
    return aux_around_box(B.inf(), B.sup(), A_all, b_all, sigma);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
