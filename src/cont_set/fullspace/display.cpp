// display - the text that describes the whole space, as CORA's fullspace.display
//
// Syntax:   std::string text = F.display();   std::cout << F;
// Inputs:   -
// Outputs:  text - its class and dimension, "fullspace" and "- dimension: n"
// See also: EmptySet::display, Interval::display
//
// CORA: contSet/@fullspace/display.m

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string Fullspace::display_impl() const {
    return "fullspace\n- dimension: " + std::to_string(n_) + "\n";
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
