// linear_reset - the constructor of a linear reset, as CORA's linearReset(A, B, c)
//
// Syntax:   LinearReset reset(A);   LinearReset reset(A, B, c);
// Inputs:   A - the state map (n+, n)
//           B - the input map (n+, m), optional
//           c - the offset (n+, 1), optional
// Outputs:  reset - x+ = A x + B u + c
// See also: evaluate
//
// CORA: hybridDynamics/@linearReset/linearReset.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/hybrid_dynamics/linear_reset/linear_reset.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

LinearReset::LinearReset(Tensor A, std::optional<Tensor> B, std::optional<Tensor> c)
    : A_(std::move(A)), B_(std::move(B)), c_(std::move(c)) {
    detail::Entry entry;
    // A a matrix, B and c with its rows
    CORA_CHECK(A_.defined() && A_.ndim() == 2, Error::wrong_input_in_constructor, [&] {
        return "LinearReset: A must be a matrix (n+, n), got the shape " + shape_text(A_.shape());
    });
    const int64_t rows = A_.shape()[0];
    CORA_CHECK(
        !B_ || (B_->ndim() == 2 && B_->shape()[0] == rows), Error::wrong_input_in_constructor, [&] {
            return "LinearReset: B must be a matrix (n+, m) with the " + std::to_string(rows) +
                   " rows of A, got the shape " + shape_text(B_->shape());
        });
    CORA_CHECK(!c_ || c_->shape() == std::vector<int64_t>({rows, 1}),
               Error::wrong_input_in_constructor, [&] {
                   return "LinearReset: c must be a column (n+, 1) with the " +
                          std::to_string(rows) + " rows of A, got the shape " +
                          shape_text(c_->shape());
               });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
