// test_linear_sys_isequal - isequal and the operators == and !=
//
// The matrices agree entrywise.
//
// CORA: contDynamics/@linearSys/isequal.m
// CORA: contDynamics/@linearSys/eq.m
// CORA: contDynamics/@linearSys/ne.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/linear_sys_steps_testing.h"

using namespace cora;
using namespace test::steps;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Equal matrices are equal, a difference of 1e-3 is seen unless tol allows it.
void systems_compare_by_their_matrices(const std::string &b) {
    const Tensor B({{1.0}, {0.5}}), C({{1.0, 0.0}});
    const LinearSys sys(diagonal(), B, C);
    check(sys.isequal(LinearSys(diagonal(), B, C)) && sys == LinearSys(diagonal(), B, C),
          b + ": the same matrices");
    const Tensor A2({{-1.0, 0.0}, {0.0, -2.001}});
    check(!sys.isequal(LinearSys(A2, B, C)) && sys != LinearSys(A2, B, C), b + ": A differs");
    check(sys.isequal(LinearSys(A2, B, C), 1e-2), b + ": ... but not by 1e-2");
    check(sys != LinearSys(diagonal(), Tensor({{1.0}, {0.6}}), C), b + ": B differs");
    check(sys != LinearSys(diagonal(), B, Tensor({{1.0, 0.1}})), b + ": C differs");
    check(sys != LinearSys(diagonal()), b + ": a system without inputs differs");
    check(LinearSys(diagonal()) == LinearSys(diagonal()), b + ": two without inputs");
    check(sys != LinearSys(Tensor({{-1.0}}), Tensor({{1.0}})), b + ": the dimension differs");
}

/// No C means C = I; CORA compares the columns of A as a set and calls permuted columns equal,
/// CORA.cpp compares entrywise (docs/cora-differences.md).
void no_output_matrix_is_the_identity_and_columns_keep_their_place(const std::string &b) {
    const Tensor I = Tensor::eye(2), B = Tensor({{1.0}, {0.0}});
    check(LinearSys(diagonal(), B) == LinearSys(diagonal(), B, I), b + ": C absent is I");
    check(LinearSys(diagonal(), B) != LinearSys(diagonal(), B, Tensor({{1.0, 0.0}, {0.0, 2.0}})),
          b + ": C = diag(1, 2) is not I");
    const Tensor A({{1.0, 2.0}, {3.0, 4.0}}), swapped({{2.0, 1.0}, {4.0, 3.0}});
    check(LinearSys(A) != LinearSys(swapped), b + ": the columns of A swapped");
}

/// A negative tol is refused; with the checks off the answer is the same.
void checks_and_their_switch(const std::string &b) {
    const LinearSys sys(diagonal());
    check(test::raises([&] { sys.isequal(sys, -1.0); }, Error::wrong_value), b + ": tol -1");
    ChecksScope off(false);
    check(sys.isequal(sys), b + ": with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        systems_compare_by_their_matrices(b);
        no_output_matrix_is_the_identity_and_columns_keep_their_place(b);
        checks_and_their_switch(b);
    });
    return test::finish("linear_sys isequal");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
