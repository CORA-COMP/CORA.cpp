// display - the text that describes a linear system, as CORA's linearSys.display
//
// Syntax:   std::cout << sys;   std::string text = sys.display();   (Python: repr(sys))
// Inputs:   sys - linear system
// Outputs:  text - the state-space equations and the matrices A, B and C
// See also: isequal
//
// CORA: contDynamics/@linearSys/display.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string LinearSys::display() const {
    detail::Entry entry;
    const LinearSys &sys = *this;
    std::string text = "Type: Linear continuous-time time-invariant system\n";
    text += sys.B() ? "State-space equation: x' = Ax + Bu\n" : "State-space equation: x' = Ax\n";
    text += "System matrix:\n" + format_matrix(sys.A(), "    ");
    if (sys.B()) text += "Input matrix:\n" + format_matrix(*sys.B(), "    ");
    if (!sys.C()) return text + "Output equation: y = x\n";
    return text + "Output equation: y = Cx\nOutput matrix:\n" + format_matrix(*sys.C(), "    ");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
