// priv_verify_energy - a checked growth bound kappa e^{rt} of e^{tA} from an energy weight
//
// A second-order system A = [[0, I, 0], [-K, -C, F], [0, 0, 0]] (h positions and velocities, z
// constant inputs) grows in no logarithmic norm, but its energy does not grow. CORALean's
// MatExp.energyWeightFloat recognises the form exactly and proposes P = blockdiag(sym(DK), D,
// beta I), D a diagonal mass that symmetrizes K; MatExp.energyGrowthFloat checks it in powers of
// two sigma ~ 1/sqrt(P_ii): P~ - aI (a = 2^-k, the first k <= 40 that passes) and
// 2rP~ - P~A~ - (P~A~)' are certified positive semi-definite by a Cholesky factor (no coupling of
// positions to the rest, so no fill), and then ||S^-1 e^{tA} S|| <= kappa e^{rt}. Bit for bit at
// nearest on the lean backend; the matrices are kept sparse, which changes no sum.
//
// Syntax:   auto w = priv_verify_energy_weight(a, n, r);
//           auto kappa = priv_verify_energy_growth(a, s, n, r);
// Inputs:   a - (n, n) system matrix, row-major
//           s - positive scales of the run
//           r - positive rate (1 / t_final)
// Outputs:  w - the number of positions and the proposed weight, empty if a has not the form
//           kappa - the factor, empty if the weight is not accepted
// See also: priv_verify_exp_ball.h, priv_verify_ra_support_function.cpp
//
// CORA.cpp only: the float definitions of CORALean's MatExp.energyWeightFloat and
// MatExp.energyGrowthFloat.
// Authors:       Claude Opus 5.5
// Written:       17-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// A sparse matrix by rows: (column, value) pairs in increasing columns, no zeros.
using EnergyRows = std::vector<std::vector<std::pair<int64_t, double>>>;

/// CORALean's MatExp.energyWeightFloat: the positions h and the weight P, if a has the form.
struct EnergyWeight {
    int64_t h = 0;
    EnergyRows P;
};

/// CORALean's MatExp.energyWeightFloat of the row-major (n, n) matrix a at the rate r.
std::optional<EnergyWeight> priv_verify_energy_weight(const std::vector<double> &a, int64_t n,
                                                      double r);

/// CORALean's MatExp.energyGrowthFloat: kappa with ||S^-1 e^{tA} S|| <= kappa e^{rt}.
std::optional<double> priv_verify_energy_growth(const std::vector<double> &a,
                                                const std::vector<double> &s, int64_t n, double r);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
