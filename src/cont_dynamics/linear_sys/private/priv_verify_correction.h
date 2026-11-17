// priv_verify_correction - the interval matrices F and G of verify's support function algorithm
//
// CORA's aux_intmat sums the Taylor terms of the correction matrices F (state) and G (input) and
// stops where a term no longer changes the sums, with no remainder. Here CORALean's sound version
// (MatExp.verifyCorrectionStateFloat, MatExp.verifyCorrectionInputFloat): the same terms and stop,
// the coefficient corrCoeffLowTight, and the remainder past the last order bounded through the
// powers-of-two scales of MatExp.balanceScalesFloat. Bit for bit at nearest.
//
// Syntax:   VerifyPowers P(A_rows, n);   auto F = priv_verify_correction_state(P, s, dt);
// Inputs:   A - row-major (n, n) system matrix
//           s - positive scales
//           dt - step size
// Outputs:  the order the sum stopped at and the bounds of the interval matrix, empty if none
// See also: priv_verify_ra_support_function.cpp
//
// CORA.cpp only: the float definitions of CORALean's MatExp.verifyCorrection*Float.
// Authors:       Claude Opus 5.5
// Written:       17-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The powers A^i / i! of CORALean's MatExp.verifyPowerFloat, row-major (n, n), kept as they
/// are asked for: each entry of X A a sum over the inner index from the last down, then / i.
class VerifyPowers {
  public:
    VerifyPowers(std::vector<double> A, int64_t n);

    int64_t n() const { return n_; }
    const std::vector<double> &A() const { return A_; }

    /// A^i / i! for i >= 1.
    const std::vector<double> &power(int i);

  private:
    int64_t n_;
    std::vector<double> A_;
    std::vector<int64_t> row_ptr_, col_idx_; // A by rows of nonzeros
    std::vector<double> val_;
    std::vector<std::vector<double>> powers_; // entry i - 1 is A^i / i!
};

/// The bounds of an interval matrix, row-major (n, n), and the order its sum stopped at.
struct VerifyCorrection {
    int order = 0;
    std::vector<double> inf, sup;
};

/// CORALean's MatExp.balanceScalesFloat: five sweeps of Osborne's balancing of |A|, powers of 2.
std::vector<double> priv_verify_balance_scales(const std::vector<double> &A, int64_t n);

/// CORALean's MatExp.scaledRemainderFloat: (s_j / s_k) tau, row-major, empty if not admissible.
std::optional<std::vector<double>> priv_verify_scaled_remainder(const std::vector<double> &A,
                                                                const std::vector<double> &s,
                                                                int64_t n, double dt, int eta);

/// CORALean's MatExp.verifyCorrectionStateFloat: F, empty if the sum does not stop by order 74.
std::optional<VerifyCorrection>
priv_verify_correction_state(VerifyPowers &P, const std::vector<double> &s, double dt);

/// CORALean's MatExp.verifyCorrectionInputFloat: G, empty if the sum does not stop by order 74.
std::optional<VerifyCorrection>
priv_verify_correction_input(VerifyPowers &P, const std::vector<double> &s, double dt);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
