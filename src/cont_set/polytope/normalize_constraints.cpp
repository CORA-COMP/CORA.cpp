// normalize_constraints - a polytope's halfspaces scaled, as CORA's normalizeConstraints
//
// The same set with its rows scaled: "b" (default; also "be") divides each row by the magnitude of
// its offset so that it becomes +-1 (an equality's 1; rows with offset 0 stay), "A" (also "Ae")
// divides each row by its Euclidean length (rows of length 0 stay). CORA applies every type to the
// inequalities and the equalities. A vertex polytope has no rows (Error::special_error).
//
// Syntax:   Pres = P.normalize_constraints();   Pres = P.normalize_constraints("A");
// Inputs:   type - "b" (default), "be", "A" or "Ae"
// Outputs:  Pres - the polytope with the scaled rows
// See also: Polytope::compact, Polytope::constraints
//
// CORA: contSet/@polytope/normalizeConstraints.m
// CORA: contSet/@polytope/private/priv_normalizeConstraints.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The rows A (..., m, n) divided by the factor (..., m, 1), and the offsets b by the same.
std::pair<Tensor, Tensor> aux_scale(const Tensor &A, const Tensor &b, const Tensor &factor) {
    return {A.divide(factor), b.divide(factor)};
}

/// The rows scaled by the magnitude of the offset: b = +-1 (CORA scales an equality by its signed
/// offset, so that it becomes 1), rows with offset 0 unchanged.
std::pair<Tensor, Tensor> aux_by_offset(const Tensor &A, const Tensor &b, bool equality) {
    const Tensor zero = b.zeros_like();
    const Tensor nonzero = b.not_equal(zero);
    const Tensor magnitude = equality ? b : b.abs();
    const Tensor factor = Tensor::where(nonzero, magnitude, b.ones_like());
    const Tensor sign = equality
                            ? b.ones_like()
                            : Tensor::where(b.greater(zero), b.ones_like(), b.ones_like() * -1.0);
    return {A.divide(factor), Tensor::where(nonzero, sign, b)};
}

/// The rows scaled to unit length, rows of length 0 (up to eps) unchanged.
std::pair<Tensor, Tensor> aux_by_length(const Tensor &A, const Tensor &b) {
    const Tensor length = A.vector_norm(Tensor::Norm::two, -1, true);
    const Tensor nonzero = length.greater(length.zeros_like() + machine_eps(A.dtype()));
    return aux_scale(A, b, Tensor::where(nonzero, length, length.ones_like()));
}

} // namespace


// ===========================================  MAIN  =========================================== //

Polytope Polytope::normalize_constraints(const std::string &type) const {
    detail::Entry entry;
    CORA_CHECK(is_h_rep(), Error::special_error, [] {
        return std::string("Halfspace representation not computed... nothing to normalize!");
    });
    CORA_CHECK(type == "b" || type == "be" || type == "A" || type == "Ae", Error::wrong_value, [&] {
        return "normalize_constraints: the type must be b, be, A or Ae, got '" + type + "'";
    });
    Parts parts;
    parts.V = V_;
    if (type == "b" || type == "be") {
        std::tie(parts.A, parts.b) = aux_by_offset(A_, b_, false);
        std::tie(parts.Ae, parts.be) = aux_by_offset(Ae_, be_, true);
    } else {
        std::tie(parts.A, parts.b) = aux_by_length(A_, b_);
        std::tie(parts.Ae, parts.be) = aux_by_length(Ae_, be_);
    }
    return Polytope(std::move(parts));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
