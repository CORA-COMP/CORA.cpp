// test_lean_linear_sys_taylor_matrices - CORA's taylorMatrices E, F and G against CORALean
//
// By default the remainder W of LinearSys (expm_remainder) and the correction matrices F and G
// (correction_matrix_state, correction_matrix_input) are CORALean's taylorMatricesFloat (the
// oracle's WCora, FCora, GCora): e^{|A| dt} by scaling and squaring at the order asked for, bit
// for bit at nearest, for random systems of two to six states at several step sizes and orders
// (1 included), a nilpotent, a zero and a stiff matrix (scales up to 1e6, |A| dt up to about 40).
// Under CORACPP_TAYLOR=entrywise the matrices are the entrywise ones at the admissible order (the
// oracle's FEntrywise, GEntrywise; null where no order admits the enclosure).
//
// ORACLE: contDynamics/@linearSys/taylorMatrices.m
// ORACLE: contDynamics/@linearSys/private/priv_expmRemainder.m
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       13-November-2026
// Last update:   16-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_closed_form_step.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/oracle/linear_sys.h"
#include "oracle_testing.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// F, G and W of the system A, CORA.cpp's against the oracle's.
void aux_compare(const std::string &name, const Tensor &A, double dt, int eta) {
    const std::optional<lean::TaylorMatrices> theirs = lean::LinearSys(A).taylor_matrices(dt, eta);
    const std::optional<EntrywiseCorrections> mine = priv_entrywise_corrections(A, dt, eta);
    test::check(mine.has_value() == theirs.has_value(), name + ": the same admissibility");
    if (!mine || !theirs) return;
    test::check(mine->order == theirs->order, name + ": the same order");
    test::oracle::check(name + " W", mine->W, theirs->W);
    test::oracle::check(name + " F", mine->F, theirs->F);
    test::oracle::check(name + " G", mine->G, theirs->G);
    // the public methods take them on a plain float64 matrix
    const LinearSys sys(A);
    test::oracle::check(name + " correction_matrix_state", sys.correction_matrix_state(dt, eta),
                        theirs->F);
    test::oracle::check(name + " correction_matrix_input", sys.correction_matrix_input(dt, eta),
                        theirs->G);
}

/// W, F and G at the order asked for, CORA.cpp's default against the oracle's taylorMatricesFloat.
void aux_compare_cora(const std::string &name, const Tensor &A, double dt, int eta) {
    const std::optional<lean::TaylorMatrices> theirs =
        lean::LinearSys(A).taylor_matrices_cora(dt, eta);
    const std::optional<EntrywiseCorrections> mine = priv_squaring_corrections(A, dt, eta);
    test::check(mine.has_value() == theirs.has_value(), name + ": the same halvings");
    if (!mine || !theirs) return;
    test::oracle::check(name + " W", mine->W, theirs->W);
    test::oracle::check(name + " F", mine->F, theirs->F);
    test::oracle::check(name + " G", mine->G, theirs->G);
    // the public methods, where the powers stay finite (else they keep CORA's expm)
    for (const Tensor &T : {mine->W, mine->F.inf(), mine->F.sup(), mine->G.inf(), mine->G.sup()})
        for (const double x : T.data())
            if (!std::isfinite(x)) return;
    const LinearSys sys(A);
    test::oracle::check(name + " expm_remainder", sys.expm_remainder(dt, eta).sup(), theirs->W);
    test::oracle::check(name + " correction_matrix_state", sys.correction_matrix_state(dt, eta),
                        theirs->F);
    test::oracle::check(name + " correction_matrix_input", sys.correction_matrix_input(dt, eta),
                        theirs->G);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    // the default: CORA's matrices with the remainder by squaring
    test::oracle::run("linearSys.taylorMatrices", [](test::oracle::Random &rnd) {
        for (const int64_t n : {2, 3, 4, 6})
            for (const int eta : {1, 2, 4, 7})
                for (const double dt : {0.01, 0.1, 0.5})
                    aux_compare_cora("cora n" + std::to_string(n) + " eta" + std::to_string(eta) +
                                         " dt" + std::to_string(dt),
                                     rnd.matrix(n, n), dt, eta);
        aux_compare_cora("cora self-test", Tensor({{-1.0, 0.0}, {0.5, -2.0}}), 0.25, 4);
        aux_compare_cora("cora nilpotent",
                         Tensor({{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}}), 0.125, 2);
        aux_compare_cora("cora zero", Tensor({{0.0, 0.0}, {0.0, 0.0}}), 0.1, 3);
        aux_compare_cora("cora fixture 9003", Tensor({{-1000.0, 0.0}, {0.5, -2.0}}), 0.015625, 4);
        // stiff: |A| dt up to about 40 (beyond about 700 e^{|A| dt} is out of the doubles' range)
        for (const auto &[scale, dt] : std::vector<std::pair<double, double>>{
                 {3.0, 0.01}, {12.0, 0.01}, {1e3, 0.01}, {1e6, 1e-5}}) {
            const Tensor A = rnd.matrix(3, 3) * scale;
            for (const int eta : {2, 4})
                aux_compare_cora("cora stiff " + std::to_string(scale) + " eta" +
                                     std::to_string(eta),
                                 A, dt, eta);
        }
    });
    // the public methods take CORALean's matrices only when asked
    setenv("CORACPP_TAYLOR", "entrywise", 1);
    test::oracle::run("linearSys.taylorMatrices", [](test::oracle::Random &rnd) {
        for (const int64_t n : {2, 3, 4, 6})
            for (const int eta : {1, 2, 4, 7})
                for (const double dt : {0.01, 0.1, 0.5})
                    aux_compare("n" + std::to_string(n) + " eta" + std::to_string(eta) + " dt" +
                                    std::to_string(dt),
                                rnd.matrix(n, n), dt, eta);
    });
    test::oracle::run("linearSys.taylorMatrices", [](test::oracle::Random &rnd) {
        // the system of the oracle's self-test, a nilpotent and a zero matrix
        aux_compare("self-test", Tensor({{-1.0, 0.0}, {0.5, -2.0}}), 0.25, 4);
        aux_compare("nilpotent", Tensor({{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}}), 0.125,
                    2);
        aux_compare("zero", Tensor({{0.0, 0.0}, {0.0, 0.0}}), 0.1, 3);
        // stiff: the order is raised past the one asked for, then none is admissible
        for (const double scale : {3.0, 12.0, 1e6}) {
            const Tensor A = rnd.matrix(3, 3) * scale;
            for (const int eta : {2, 4})
                aux_compare("stiff " + std::to_string(scale) + " eta" + std::to_string(eta), A, 1.0,
                            eta);
        }
    });
    return test::finish("lean linearSys taylorMatrices");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
