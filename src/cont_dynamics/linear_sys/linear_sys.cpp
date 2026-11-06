// linear_sys - the constructors of a linear system, checked, as CORA's linearSys constructor
//
// Syntax:   LinearSys sys(A);   LinearSys sys(A, B);   LinearSys sys(A, B, C);
// Inputs:   A - (..., n, n);  B - (n, m);  C - (p, n), optional
// Outputs:  sys - the system x' = A x + B u, y = C x
// See also: reach, simulate, verify
//
// CORA: contDynamics/@linearSys/linearSys.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether A is a square matrix (..., n, n).
bool aux_square(const Tensor &A) {
    const std::vector<int64_t> s = A.shape();
    return s.size() >= 2 && s[s.size() - 1] == s[s.size() - 2];
}

} // namespace


// ===========================================  MAIN  =========================================== //

/// x' = A x: A square.
LinearSys::LinearSys(Tensor A) : A_(std::move(A)) {
    detail::Entry entry;
    CORA_CHECK(A_.defined() && aux_square(A_), Error::wrong_input_in_constructor, [&] {
        return "LinearSys: A must be a square matrix (..., n, n), got the shape " +
               shape_text(A_.shape());
    });
}

LinearSys::LinearSys(Tensor A, Tensor B, std::optional<Tensor> C)
    : A_(std::move(A)), B_(std::move(B)), C_(std::move(C)) {
    detail::Entry entry;
    // A square, B with the rows of A, C with its columns
    CORA_CHECK(A_.defined() && aux_square(A_), Error::wrong_input_in_constructor, [&] {
        return "LinearSys: A must be a square matrix (..., n, n), got the shape " +
               shape_text(A_.shape());
    });
    CORA_CHECK(
        B_->shape().size() == 2 && B_->shape()[0] == dim(), Error::wrong_input_in_constructor, [&] {
            return "LinearSys: B must be a matrix (n, m) with the n = " + std::to_string(dim()) +
                   " rows of A, got the shape " + shape_text(B_->shape());
        });
    CORA_CHECK(!C_ || (C_->shape().size() == 2 && C_->shape()[1] == dim()),
               Error::wrong_input_in_constructor, [&] {
                   return "LinearSys: C must be a matrix (p, n) with the n = " +
                          std::to_string(dim()) + " columns of A, got the shape " +
                          shape_text(C_->shape());
               });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
