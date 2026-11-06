// test_linear_sys_validate_step - validate_step: the checks that every step operation shares
//
// The set must be in the state space, the step positive and the series at least one term long.
//
// CORA.cpp only: tests validate_step, which has no CORA counterpart (validateOptions)
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

/// A set of the wrong dimension, a step that is not positive and a series without terms are
/// refused.
void wrong_arguments_are_refused(const std::string &b) {
    const LinearSys sys(diagonal());
    sys.validate_step("op", initial_set(), kStep, kTerms);
    check(test::raises([&] { sys.validate_step("op", origin(3), kStep, kTerms); },
                       Error::dimension_mismatch),
          b + ": a set of three dimensions");
    check(test::raises([&] { sys.validate_step("op", initial_set(), 0.0, kTerms); },
                       Error::wrong_value),
          b + ": time_step 0");
    check(
        test::raises([&] { sys.validate_step("op", initial_set(), kStep, 0); }, Error::wrong_value),
        b + ": no terms");
    // the message names the operation that asked
    try {
        sys.validate_step("one_step", initial_set(), kStep, 0);
    } catch (const Error &e) {
        check(std::string(e.what()).find("one_step") != std::string::npos, b + ": the message");
    }
    // the checks switch off for a caller that does not check
    ChecksScope off(false);
    sys.validate_step("op", origin(3), 0.0, 0);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(wrong_arguments_are_refused);
    return test::finish("linear_sys validate_step");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
