// display - the text that describes a zonotope, as CORA's zonotope.display
//
// Syntax:   std::string text = Z.display();   std::cout << Z;
// Inputs:   Z - zonotope
// Outputs:  text - its dimension, center and generators (a batch: only its shape)
// See also: interval display, global/format.h

#include "contSet/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {


// ===========================================  MAIN  =========================================== //

std::string Zonotope::display() const {
    std::string text = "zonotope\n";
    text += "- dimension: " + std::to_string(dim()) + "\n";
    text += "- center:\n" + formatMatrix(c, "    ");
    const std::vector<int64_t> shape = G.shape();
    text += "- generators (" + std::to_string(shape.back()) + "):\n" + formatMatrix(G, "    ");
    return text;
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
