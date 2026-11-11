// display - the text of a polynomial zonotope, as CORA's polyZonotope.display
//
// The dimension, the center and the generators with their exponents.
//
// Syntax:   text = pZ.display();
// Inputs:   -
// Outputs:  text
// See also: -
//
// CORA: contSet/@polyZonotope/display.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string PolyZonotope::display_impl() const {
    std::string text = "polyZonotope\n- dimension: " + std::to_string(dim()) + "\n";
    text += "- c:\n" + format_matrix(c_, "    ") + "- G:\n" + format_matrix(G_, "    ");
    text += "- GI:\n" + format_matrix(GI_, "    ") + "- E (one row per generator):\n";
    for (const std::vector<int> &col : E_) {
        text += "    ";
        for (const int e : col) text += std::to_string(e) + " ";
        text += "\n";
    }
    text += "- id:";
    for (const int64_t v : id_) text += " " + std::to_string(v);
    return text + "\n";
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
