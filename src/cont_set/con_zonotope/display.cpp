// display - the text that describes a constrained zonotope, as CORA's conZonotope.display
//
// Syntax:   std::string text = cz.display();   std::cout << cz;
// Inputs:   cz - constrained zonotope
// Outputs:  text - its dimension, center, generators and constraints (a batch: only the shapes)
// See also: Zonotope display, src/global/format.h
//
// CORA: contSet/@conZonotope/display.m
// CORA authors: Dmitry Grebenyuk, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string ConZonotope::display_impl() const {
    std::string text = "constrained zonotope\n";
    text += "- dimension: " + std::to_string(dim()) + "\n";
    text += "- center:\n" + format_matrix(c_.expand_dims(-1), "    ");
    text += "- generators (" + std::to_string(G_.size(-2)) + "):\n" +
            format_matrix(G_.matrix_transpose(), "    ");
    text += "- constraint matrix A (" + std::to_string(A().shape()[A().ndim() - 2]) + " rows):\n";
    text += format_matrix(A(), "    ");
    return text + "- constraint offset b:\n" + format_matrix(priv_b_column(*this), "    ");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
