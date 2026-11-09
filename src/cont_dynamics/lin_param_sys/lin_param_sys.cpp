// lin_param_sys - the constructor of a linear parametric system, as CORA's linParamSys
//
// Syntax:   LinParamSys sys(A);   LinParamSys sys(A, B, c, ParamType::varying);
// Inputs:   A - the matrix zonotope of the system matrices (n, n)
//           B - the input matrix (n, m), optional
//           c - the constant offset (n, 1), optional (default 0)
//           type - ParamType::constant (CORA's 'constParam', default) or ParamType::varying
// Outputs:  sys - x' = A x + B u + c
// See also: reach, init_reach, post
//
// CORA: contDynamics/@linParamSys/linParamSys.m
// CORA authors: Matthias Althoff
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/lin_param_sys/lin_param_sys.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

LinParamSys::LinParamSys(MatZonotope A, std::optional<Tensor> B, std::optional<Tensor> c,
                         ParamType type)
    : A_(std::move(A)), B_(std::move(B)), type_(type) {
    detail::Entry entry;
    // one square set of matrices, B with its rows and c a column of them
    CORA_CHECK(A_.center().ndim() == 2 && A_.rows() == A_.cols() && A_.rows() > 0,
               Error::wrong_input_in_constructor, [&] {
                   return "LinParamSys: A must be one square matrix zonotope (n, n), got the "
                          "center shape " +
                          shape_text(A_.center().shape());
               });
    const int64_t n = A_.rows();
    CORA_CHECK(!B_ || (B_->ndim() == 2 && B_->shape()[0] == n), Error::wrong_input_in_constructor,
               [&] {
                   return "LinParamSys: B must be a matrix (n, m) with the " + std::to_string(n) +
                          " rows of A, got the shape " + shape_text(B_->shape());
               });
    CORA_CHECK(!c || c->shape() == std::vector<int64_t>({n, 1}), Error::wrong_input_in_constructor,
               [&] {
                   return "LinParamSys: c must be a column (n, 1) with the " + std::to_string(n) +
                          " rows of A, got the shape " + shape_text(c->shape());
               });
    c_ = c ? *c : A_.center().zeros_like({n, 1});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
