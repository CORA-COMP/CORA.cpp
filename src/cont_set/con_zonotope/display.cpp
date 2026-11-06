// display - the text that describes a constrained zonotope, as CORA's conZonotope.display
//
// Syntax:   std::string text = cz.display();   std::cout << cz;
// Inputs:   cz - constrained zonotope
// Outputs:  text - its dimension, center, generators and constraints (a batch: only the shapes)
// See also: Zonotope display, src/global/format.h
//
// CORA: contSet/@conZonotope/display.m
// CORA authors: Dmitry Grebenyuk, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string ConZonotope::display_impl() const {
    std::string text = "constrained zonotope\n";
    text += "- dimension: " + std::to_string(dim()) + "\n";
    text += "- center:\n" + format_matrix(c_, "    ");
    text +=
        "- generators (" + std::to_string(G_.shape().back()) + "):\n" + format_matrix(G_, "    ");
    text += "- constraint matrix A (" + std::to_string(A_.shape()[A_.ndim() - 2]) + " rows):\n";
    text += format_matrix(A_, "    ");
    return text + "- constraint offset b:\n" + format_matrix(b_, "    ");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
