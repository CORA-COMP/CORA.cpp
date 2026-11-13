// test_lean_linear_sys_taylor_matrices - CORA's taylorMatrices F and G against CORALean
//
// The correction matrices F and G of LinearSys (correction_matrix_state, correction_matrix_input)
// are CORALean's entrywise matrices (the oracle's linearSys.taylorMatrices: FEntrywise,
// GEntrywise, with the remainder W and the order eta_e) bit for bit at nearest, for random
// systems of two to six states at several step sizes and orders, a nilpotent, a zero and a stiff
// matrix, and where no order admits the enclosure (the oracle answers null, the entrywise matrices
// are not served). CORA.cpp's priv_entrywise_corrections is compared, for W as well, and the public
// methods under CORACPP_TAYLOR=entrywise.
//
// ORACLE: contDynamics/@linearSys/taylorMatrices.m
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_closed_form_step.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/oracle/linear_sys.h"
#include "oracle_testing.h"

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

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
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
