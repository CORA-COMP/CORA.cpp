// display - the text that describes a polytope, as CORA's polytope.display
//
// Syntax:   std::string text = P.display();   std::cout << P;
// Inputs:   P - polytope
// Outputs:  text - its dimension, its vertices and its inequality and equality constraints, as far
//           as they are held (a batch: only the shapes). CORA also lists the properties it has
//           found out; here they are computed when asked for
// See also: Zonotope display, src/global/format.h
//
// CORA: contSet/@polytope/display.m
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string Polytope::display_impl() const {
    std::string text = "polytope\n";
    text += "- dimension: " + std::to_string(dim()) + "\n";
    if (is_v_rep()) {
        text +=
            "- vertices (" + std::to_string(V_.shape().back()) + "):\n" + format_matrix(V_, "    ");
    } else {
        text += "- vertices: (not computed)\n";
    }
    if (is_h_rep()) {
        const std::string rows = std::to_string(A_.shape()[A_.ndim() - 2]);
        const std::string rows_eq = std::to_string(Ae_.shape()[Ae_.ndim() - 2]);
        text +=
            "- inequality constraints A x <= b (" + rows + " rows):\n" + format_matrix(A_, "    ");
        text += format_matrix(b_, "    ");
        text += "- equality constraints Ae x = be (" + rows_eq + " rows):\n" +
                format_matrix(Ae_, "    ");
        text += format_matrix(be_, "    ");
    } else {
        text += "- constraints: (not computed)\n";
    }
    return text;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
