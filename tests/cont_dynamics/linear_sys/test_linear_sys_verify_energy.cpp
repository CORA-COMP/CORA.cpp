// test_linear_sys_verify_energy - the energy weight of a second-order system and its growth bound
//
// A beam of masses on springs, A = [[0, I, 0], [-K, -C, F], [0, 0, 0]]: no logarithmic norm bounds
// its growth, but the energy does not grow. The weight P of priv_verify_energy_weight must have the
// form blockdiag(sym(DK), D, beta I) (positive on the diagonal of the velocities), the factor of
// priv_verify_energy_growth must be at least 1 (e^{0 A} = I), and a matrix off the form has no
// weight. The values against CORALean's float definitions are checked in the oracle test
// test_lean_linear_sys_verify_support.
//
// CORA.cpp only: no CORA counterpart.
// Authors:       Claude Opus 5.5
// Written:       19-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_verify_correction.h"
#include "cont_dynamics/linear_sys/private/priv_verify_energy.h"
#include "testing.h"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A beam of h masses m on springs k (fixed ends), damping alpha + beta K, and z inputs pushing the
/// last mass: [[0, I, 0], [-K, -C, F], [0, 0, 0]] with K = M^-1 K_s, row-major (2h + z, 2h + z).
std::vector<double> aux_beam(int64_t h, double k, const std::vector<double> &m, double alpha,
                             double beta, int64_t z) {
    const int64_t n = 2 * h + z;
    std::vector<double> a(std::size_t(n) * n, 0.0);
    auto at = [&](int64_t i, int64_t j) -> double & { return a[std::size_t(i * n + j)]; };
    for (int64_t i = 0; i < h; ++i) {
        at(i, h + i) = 1.0;
        const double mi = m[std::size_t(i) % m.size()];
        for (int64_t j = std::max<int64_t>(0, i - 1); j <= std::min(h - 1, i + 1); ++j) {
            const double kij = (i == j ? 2.0 * k : -k) / mi;
            at(h + i, j) = -kij;
            at(h + i, h + j) = -(i == j ? alpha + beta * kij : beta * kij);
        }
    }
    for (int64_t q = 0; q < z; ++q) at(2 * h - 1 - q, 2 * h + q) = 1.0 / double(q + 1);
    return a;
}

/// The weight and the factor of a beam at t_final: the weight has h positions and a positive
/// diagonal, the factor is at least 1.
void aux_beam_check(const std::string &name, int64_t h, double k, double alpha, double beta,
                    int64_t z, double t_final) {
    const int64_t n = 2 * h + z;
    const std::vector<double> a = aux_beam(h, k, {1.0, 2.0, 1.0, 3.0}, alpha, beta, z);
    const double r = 1.0 / t_final;
    const std::optional<EnergyWeight> w = priv_verify_energy_weight(a, n, r);
    check(w.has_value(), name + ": a weight");
    if (!w) return;
    check(w->h == h, name + ": the positions");
    check(w->P.size() == std::size_t(n), name + ": a row per state");
    bool positive = true;
    for (int64_t i = 0; i < n; ++i)
        for (const auto &[j, v] : w->P[std::size_t(i)])
            if (i == j && !(v > 0)) positive = false;
    check(positive, name + ": a positive diagonal");
    const std::optional<double> kappa =
        priv_verify_energy_growth(a, priv_verify_balance_scales(a, n), n, r);
    check(kappa.has_value() && *kappa >= 1.0 && std::isfinite(*kappa), name + ": a factor >= 1");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    aux_beam_check("beam 3", 3, 100.0, 0.1, 0.01, 1, 1.0);
    aux_beam_check("beam 4 no input", 4, 50.0, 0.05, 0.02, 0, 2.0);
    aux_beam_check("beam 2 two inputs", 2, 1e4, 1.0, 1e-3, 2, 0.01);

    // undamped: no decay, but the weight still bounds the growth over one time unit
    const std::vector<double> undamped = aux_beam(3, 10.0, {1.0, 2.0, 1.0, 3.0}, 0.0, 0.0, 1);
    const std::optional<double> factor =
        priv_verify_energy_growth(undamped, priv_verify_balance_scales(undamped, 7), 7, 1.0);
    check(factor.has_value() && *factor >= 1.0, "undamped: a factor >= 1");

    // one ulp off the form: no weight; a heat chain has no positions
    std::vector<double> off = aux_beam(3, 100.0, {1.0, 2.0, 1.0, 3.0}, 0.1, 0.01, 1);
    off[3] = std::nextafter(1.0, 2.0);
    check(!priv_verify_energy_weight(off, 7, 1.0).has_value(), "one ulp off: no weight");
    const std::vector<double> chain{-1, 1, 0, 1, -2, 1, 0, 1, -1};
    check(!priv_verify_energy_weight(chain, 3, 1.0).has_value(), "heat: no weight");
    return test::finish("linear_sys verify energy");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
