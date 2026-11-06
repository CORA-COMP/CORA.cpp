// isequal - whether two linear systems are equal, as CORA's isequal (and eq, ne)
//
// The matrices A, B and C agree entrywise up to tol; without C the output is the state, so C
// absent equals C = I. CORA compares the columns of A and B as sets in any order (compareMatrices),
// which calls two systems with permuted state columns equal; CORA.cpp compares entrywise.
//
// Syntax:   bool res = sys.isequal(other);   sys.isequal(other, 1e-10);   sys == other
// Inputs:   other - linear system
//           tol - relative or absolute tolerance, default: the repository tolerance
// Outputs:  res - whether the systems are equal (operator== takes the default tol)
// See also: operator<<
//
// CORA: contDynamics/@linearSys/isequal.m
// CORA: contDynamics/@linearSys/eq.m
// CORA: contDynamics/@linearSys/ne.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/within_tol.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether a and b have one shape and agree entrywise up to tol.
bool aux_same(const Tensor &a, const Tensor &b, double tol) {
    return std::vector<int64_t>(a.shape()) == std::vector<int64_t>(b.shape()) &&
           within_tol(a, b, tol).min().item() != 0;
}

/// Whether two optional matrices are both absent or equal.
bool aux_same(const std::optional<Tensor> &a, const std::optional<Tensor> &b, double tol) {
    return a.has_value() == b.has_value() && (!a || aux_same(*a, *b, tol));
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool LinearSys::isequal(const LinearSys &other, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, A_.dtype(), "isequal");
    // an absent output matrix is the identity
    const auto output = [](const LinearSys &s) {
        return s.C_ ? s.C_ : std::optional(s.A_.eye_like());
    };
    return aux_same(A_, other.A_, tol) && aux_same(B_, other.B_, tol) &&
           aux_same(output(*this), output(other), tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
