// test_lean_linear_sys_verify_correction - F and G of verify's support function algorithm
//
// The balancing scales and the interval matrices F (state) and G (input) that verify's support
// function algorithm takes from priv_verify_correction, against CORALean's
// MatExp.balanceScalesFloat, verifyCorrectionStateFloat and verifyCorrectionInputFloat, bit for bit
// at nearest: random systems of two to six states at several step sizes, a stiff oscillator whose
// norm only the scales bring down, an upper-triangular and a nilpotent matrix, and a step too large
// for the sums to stop (both empty). Tagged CORA, not ORACLE: the rest of the algorithm is not yet
// bit for bit.
//
// CORA: contDynamics/@linearSys/private/priv_verifyRA_supportFunc.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       17-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_verify_correction.h"
#include "cora/global/oracle/linear_sys.h"
#include "oracle_testing.h"

#include <optional>
#include <string>
#include <vector>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// One matrix of CORA.cpp against the oracle's: the same order and the same bounds.
void aux_check_matrix(const std::string &name, const std::optional<VerifyCorrection> &mine,
                      const std::optional<std::pair<int64_t, lean::Interval>> &theirs, int64_t n) {
    test::check(mine.has_value() == theirs.has_value(), name + ": both stop or neither");
    if (!mine || !theirs) return;
    test::check(mine->order == theirs->first, name + ": the same order");
    const Interval M(Tensor::from_data(mine->inf, {n, n}), Tensor::from_data(mine->sup, {n, n}));
    test::oracle::check(name, M, theirs->second);
}

/// The scales, F and G of the system A at dt, CORA.cpp's against the oracle's.
void aux_compare(const std::string &name, const Tensor &A, double dt) {
    const int64_t n = A.shape()[0];
    const lean::VerifyCorrection theirs = lean::LinearSys(A).verify_correction(dt);
    const std::vector<double> s = priv_verify_balance_scales(A.data(), n);
    test::oracle::check(name + " scales", Tensor::from_data(s, {n, 1}), theirs.scales);
    VerifyPowers P(A.data(), n);
    aux_check_matrix(name + " F", priv_verify_correction_state(P, s, dt), theirs.F, n);
    aux_check_matrix(name + " G", priv_verify_correction_input(P, s, dt), theirs.G, n);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("linearSys.verifyCorrection", [](test::oracle::Random &rnd) {
        for (const int64_t n : {2, 3, 4, 6})
            for (const double dt : {0.01, 0.1, 0.5})
                aux_compare("n" + std::to_string(n) + " dt" + std::to_string(dt), rnd.matrix(n, n),
                            dt);
        // the fixtures 9302 and 9303 of the oracle's self-test
        aux_compare("fixture 9302", Tensor({{-1.0, 0.0}, {0.5, -2.0}}), 0.25);
        aux_compare("fixture 9303", Tensor({{0.0, -10000.0}, {1.0, -1.0}}), 1.0 / 1024);
        // a stiff oscillator: ||A dt|| is about 10^3, the balanced norm about 1
        aux_compare("oscillator",
                    Tensor({{0.0, 1.0, 0.0, 0.0},
                            {-1e8, -10.0, 1e8, 0.0},
                            {0.0, 0.0, 0.0, 1.0},
                            {1e8, 0.0, -2e8, -10.0}}),
                    1e-5);
        aux_compare("triangular", Tensor({{-1.0, 2.0, 0.0}, {0.0, -3.0, 1.0}, {0.0, 0.0, -0.5}}),
                    0.2);
        aux_compare("nilpotent", Tensor({{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}}),
                    0.125);
        // too large a step: the terms grow past order 74, neither sum stops
        aux_compare("diverging", Tensor({{-50.0, 3.0}, {1.0, -40.0}}), 2.0);
    });
    return test::finish("lean linearSys verifyCorrection");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
