// linear_sys - the constructors of a linear system, checked, as CORA's linearSys constructor
//
// A sparse A is kept as it is; A() forms its dense matrix (like B) once, on first use, and the
// copies of the system share it.
//
// Syntax:   LinearSys sys(A);   LinearSys sys(A, B);   LinearSys sys(A, B, C);
// Inputs:   A - (..., n, n), or a SparseMatrix (n, n) with B
//           B - (n, m)
//           C - (p, n), optional
// Outputs:  sys - the system x' = A x + B u, y = C x
// See also: reach, simulate, verify
//
// CORA: contDynamics/@linearSys/linearSys.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The checks of B (n, m) and C (p, n) against the n states.
void aux_check_input_output(const Tensor &B, const std::optional<Tensor> &C, int64_t n) {
    detail::Validation entry;
    CORA_CHECK(B.shape().size() == 2 && B.shape()[0] == n, Error::wrong_input_in_constructor, [&] {
        return "LinearSys: B must be a matrix (n, m) with the n = " + std::to_string(n) +
               " rows of A, got the shape " + shape_text(B.shape());
    });
    CORA_CHECK(!C || (C->shape().size() == 2 && C->shape()[1] == n),
               Error::wrong_input_in_constructor, [&] {
                   return "LinearSys: C must be a matrix (p, n) with the n = " + std::to_string(n) +
                          " columns of A, got the shape " + shape_text(C->shape());
               });
}

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
    aux_check_input_output(*B_, C_, dim());
}

LinearSys::LinearSys(SparseMatrix A, Tensor B, std::optional<Tensor> C)
    : A_sparse_(std::make_shared<const SparseMatrix>(std::move(A))),
      dense_A_(std::make_shared<DenseA>()), B_(std::move(B)), C_(std::move(C)) {
    detail::Entry entry;
    CORA_CHECK(A_sparse_->rows() == A_sparse_->cols(), Error::wrong_input_in_constructor, [&] {
        return "LinearSys: A must be square, got " + std::to_string(A_sparse_->rows()) + " x " +
               std::to_string(A_sparse_->cols());
    });
    aux_check_input_output(*B_, C_, dim());
}

const Tensor &LinearSys::A() const {
    if (!A_sparse_) return A_;
    // the first caller forms the dense matrix; the others wait for it
    std::call_once(dense_A_->once, [this] { dense_A_->A = A_sparse_->dense_like(*B_); });
    return dense_A_->A;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
