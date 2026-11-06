// display - the text that describes a capsule, as CORA's capsule.display
//
// Syntax:   std::string text = C.display();   std::cout << C;
// Inputs:   C - capsule
// Outputs:  text - its dimension, center, generator and radius (a batch: only the shapes)
// See also: ConZonotope display, src/global/format.h
//
// CORA: contSet/@capsule/display.m

#include "cora/cont_set/capsule/capsule.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string Capsule::display_impl() const {
    std::string text = "capsule\n";
    text += "- dimension: " + std::to_string(dim()) + "\n";
    text += "- center:\n" + format_matrix(c_, "    ");
    text += "- generator:\n" + format_matrix(g_, "    ");
    return text + "- radius:\n" + format_matrix(r_, "    ");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
