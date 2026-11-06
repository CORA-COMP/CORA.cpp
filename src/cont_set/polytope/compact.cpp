// compact - fewer rows or vertices for the same polytope, as CORA's polytope.compact
//
// Syntax:   Pres = P.compact();   Pres = P.compact("all", tol);   Pres = P.compact("V");
// Inputs:   method - "all" (default): a minimal halfspace form; "A" the inequalities only (pairs
//                    become equalities first); "Ae" the equalities (aligned ones merged);
//                    "aligned" the aligned rows; "zeros" the rows 0 <= b and 0 = b; "AtoAe" pairs
//                    a x <= c, -a x <= -c become the equality a x = c; "V" the vertices that are
//                    not extreme (needs the vertex form)
//           tol - what counts as zero and as aligned; default: the repository tolerance
//                 (docs/architecture.md). "Ae" and the equalities of "aligned" compare up to
//                 machine precision, as CORA's (its tol is not used there)
// Outputs:  Pres - the halfspace methods normalize every row to unit length and return the empty
//           set or R^n if the rows say so; a polytope that also holds vertices keeps them.
//           "V" keeps the extreme vertices in their order (in 1D the least and the greatest, as
//           two columns). Not a batch
// See also: Polytope::constraints, Zonotope::compact
//
// CORA: contSet/@polytope/compact_.m
// CORA: contSet/@polytope/private/priv_compact_V.m
// CORA: contSet/@polytope/private/priv_copyProperties.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

const char *const kMethods[] = {"all", "A", "Ae", "aligned", "zeros", "AtoAe", "V"};

/// Applies the halfspace method to H (normalized); true if H is found to be empty.
bool aux_apply(const std::string &method, HostH &H, double tol) {
    if (method == "zeros") return priv_compact_zeros(H, tol);
    if (method == "aligned") {
        priv_compact_aligned_inequalities(H, tol);
        return priv_compact_aligned_equalities(H);
    }
    if (method == "AtoAe") {
        priv_compact_to_equality(H, tol);
        return false;
    }
    if (method == "A") {
        priv_compact_to_equality(H, tol);
        priv_compact_aligned_inequalities(H, tol);
        // the inequalities alone, CORA hands the equalities to nothing here
        HostH inequalities = H;
        inequalities.Ae.clear();
        inequalities.be.clear();
        const bool empty = priv_compact_nd(inequalities, tol);
        H.A = inequalities.A;
        H.b = inequalities.b;
        return empty;
    }
    if (method == "Ae") return priv_compact_aligned_equalities(H);
    if (method == "all") {
        bool minimal = false;
        return priv_compact_all(H, tol, minimal);
    }
    CORA_FAIL(Error::wrong_value, "compact: unknown method '" + method + "'");
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The compaction ----------------------------------------------------------------------------------

Polytope Polytope::compact(const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, (is_h_rep() ? A_ : V_).dtype(), "compact");
    CORA_CHECK(std::find(std::begin(kMethods), std::end(kMethods), method) != std::end(kMethods),
               Error::wrong_value, [&] {
                   return "compact: the method must be 'all', 'A', 'Ae', 'aligned', 'zeros', "
                          "'AtoAe' or 'V', got '" +
                          method + "'";
               });
    if (method == "V") {
        CORA_CHECK(is_v_rep(), Error::not_supported, [] {
            return std::string("compact: the vertex representation is not provided");
        });
        const Rows points = priv_host_points(V_);
        Polytope out = *this;
        if (points.size() <= 1) return out;
        // in 1D the least and the greatest vertex, else the extreme ones
        out.V_ = dim() == 1 ? Tensor::concat({V_.min(-1, true), V_.max(-1, true)}, -1)
                            : V_.take(priv_extreme_points(points), -1);
        return out;
    }
    const int64_t n = dim();
    HostH H = priv_host_h(constraints());
    priv_normalize_rows(H);
    if (aux_apply(method, H, tol)) return empty(n);
    // no row is left: R^n
    if (H.A.empty() && H.Ae.empty()) return inf(n);
    const Constraints C = priv_tensors_h(is_h_rep() ? A_ : V_, H);
    Polytope out(C.A, C.b, C.Ae, C.be);
    // CORA copies what the input knows, the vertices among it
    if (is_v_rep()) out.V_ = V_;
    return out;
}

// The interface of the set ------------------------------------------------------------------------

std::optional<std::vector<std::string>> Polytope::compact_methods_impl() const {
    return std::vector<std::string>(std::begin(kMethods), std::end(kMethods));
}

Set Polytope::compact_impl(const std::optional<std::string> &method, Tol tol) const {
    return compact(method.value_or("all"), tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
