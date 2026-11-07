// test_linear_sys_is_canonical_form - is_canonical_form: B and C are identity matrices
//
// Exactly the identity counts: a difference of 1e-15 does not, and a system without B never does.
//
// CORA: contDynamics/@linearSys/isCanonicalForm.m
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

/// B and C must be identities (exactly), without B the system is not canonical, without C it is.
void identity_matrices_are_canonical(const std::string &b) {
    const Tensor I = Tensor::eye(2), P({{1.0, 1e-15}, {0.0, 1.0}});
    check(LinearSys(diagonal(), I).is_canonical_form(), b + ": B = I, no C");
    check(LinearSys(diagonal(), I, I).is_canonical_form(), b + ": B = I, C = I");
    check(!LinearSys(diagonal(), P).is_canonical_form(), b + ": B differs by 1e-15");
    check(!LinearSys(diagonal(), I, P).is_canonical_form(), b + ": C differs by 1e-15");
    check(!LinearSys(diagonal()).is_canonical_form(), b + ": no inputs");
    check(!LinearSys(diagonal(), Tensor({{1.0}, {0.0}})).is_canonical_form(), b + ": B is 2 x 1");
    check(!LinearSys(diagonal(), I, Tensor({{1.0, 0.0}})).is_canonical_form(), b + ": C is 1 x 2");
    // there is nothing to check: the same answer with the checks off
    ChecksScope off(false);
    check(LinearSys(diagonal(), I, I).is_canonical_form(), b + ": with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { identity_matrices_are_canonical(b); });
    return test::finish("linear_sys is_canonical_form");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
