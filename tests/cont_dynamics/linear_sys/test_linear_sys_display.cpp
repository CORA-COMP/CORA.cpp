// test_linear_sys_display - display and operator<<
//
// The text names the equations and the matrices of the system.
//
// CORA: contDynamics/@linearSys/display.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/linear_sys_steps_testing.h"

#include <sstream>

using namespace cora;
using namespace test::steps;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

bool has(const std::string &text, const std::string &part) {
    return text.find(part) != std::string::npos;
}

/// The text names the equation and every matrix the system has, and operator<< prints it.
void the_text_lists_the_matrices(const std::string &b) {
    const std::string autonomous = LinearSys(diagonal()).display();
    check(has(autonomous, "x' = Ax\n") && has(autonomous, "System matrix") &&
              has(autonomous, "-2") && has(autonomous, "Output equation: y = x"),
          b + ": an autonomous system");
    const LinearSys sys(diagonal(), Tensor({{1.0}, {0.5}}), Tensor({{1.0, 0.0}}));
    const std::string text = sys.display();
    check(has(text, "x' = Ax + Bu") && has(text, "Input matrix") && has(text, "0.5") &&
              has(text, "Output matrix") && has(text, "y = Cx"),
          b + ": a system with inputs and outputs");
    std::ostringstream out;
    out << sys;
    check(out.str() == text, b + ": operator<<");
    // with the checks off the text is the same
    ChecksScope off(false);
    check(sys.display() == text, b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { the_text_lists_the_matrices(b); });
    return test::finish("linear_sys display");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
