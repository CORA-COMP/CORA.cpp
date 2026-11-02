// display - the text that describes a zonotope, as CORA's zonotope.display
//
// Syntax:   std::string text = Z.display();   std::cout << Z;
// Inputs:   Z - zonotope
// Outputs:  text - its dimension, center and generators (a batch: only its shape)
// See also: interval display, src/global/format.h
//
// CORA: contSet/@zonotope/display.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string Zonotope::display_impl() const {
    std::string text = "zonotope\n";
    text += "- dimension: " + std::to_string(dim()) + "\n";
    text += "- center:\n" + format_matrix(c_, "    ");
    const std::vector<int64_t> shape = G_.shape();
    text += "- generators (" + std::to_string(shape.back()) + "):\n" + format_matrix(G_, "    ");
    return text;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
