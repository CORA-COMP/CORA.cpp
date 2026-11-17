// priv_verify_exp_ball - e^{A dt} as a matrix and a balanced radius, and the growth of e^{tA}
//
// CORA's priv_verifyRA_supportFunc propagates its directions by a point expm, which encloses
// nothing. Here CORALean's MatExp.expBallFloat: the Taylor series of A dt / 2^sigma (sigma halvings
// until the balanced norm is at most 4, the order the first from 6 whose remainder bound is at
// most 2^-80), squared sigma times, with the radius r of ||S^-1 (e^{A dt} - D) S|| <= r carried
// through the squarings; and the bounds MatExp.step_scaledNormUpFloat (the balanced norm) and
// MatExp.logNormUpFloat (the balanced logarithmic norm, ||S^-1 e^{tA} S|| <= e^{t mu}). Bit for bit
// at nearest on the lean backend.
//
// Syntax:   auto ball = priv_verify_exp_ball(A, s, dt);
//           double mu = priv_verify_log_norm_up(a, s, n);
// Inputs:   A - (n, n) system matrix (a, row-major)
//           s - positive scales
//           dt - step size
// Outputs:  ball - the matrix D and the radius r, empty if no halving count or order is found
// See also: priv_verify_ra_support_function.cpp, priv_verify_correction.h
//
// CORA.cpp only: the float definitions of CORALean's MatExp.expBallFloat,
// step_scaledNormUpFloat and logNormUpFloat.
// Authors:       Claude Opus 5.5
// Written:       17-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/tensor/tensor.h"

#include <cstdint>
#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// e^{A dt} within the balanced radius r of D.
struct VerifyExpBall {
    Tensor D;
    double r = 0;
};

/// CORALean's MatExp.step_scaledNormUpFloat: max(0, max_j sum_k |m_jk| s_k / s_j), row-major
/// (n, n).
double priv_verify_scaled_norm_up(const std::vector<double> &m, const std::vector<double> &s,
                                  int64_t n);

/// CORALean's MatExp.logNormUpFloat: max_j (a_jj + sum_{k != j} |a_jk| s_k / s_j).
double priv_verify_log_norm_up(const std::vector<double> &a, const std::vector<double> &s,
                               int64_t n);

/// CORALean's MatExp.expBallFloat of A at dt with the scales s.
std::optional<VerifyExpBall> priv_verify_exp_ball(const Tensor &A, const std::vector<double> &s,
                                                  double dt);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
