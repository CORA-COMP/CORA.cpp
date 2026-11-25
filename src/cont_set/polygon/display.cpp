// display - the text of a polygon, as CORA's display
//
// The dimension, the number of pieces and the vertices of each.
//
// Syntax:   std::string text = pgon.display();
// Inputs:   -
// Outputs:  text - the description
// See also: pieces
//
// CORA: contSet/@polygon/display.m
// CORA authors: Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polygon/polygon.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string Polygon::display_impl() const {
    std::string text = "polygon\n- dimension: 2\n- pieces: " + std::to_string(num_pieces()) + "\n";
    for (std::size_t i = 0; i < pieces_.size(); ++i) {
        // the vertices of the piece, one per line
        text += "- piece " + std::to_string(i + 1) + ":\n";
        const std::vector<double> v = pieces_[i].V().data();
        for (std::size_t l = 0; 2 * l < v.size(); ++l)
            text += "    " + std::to_string(v[2 * l]) + "  " + std::to_string(v[2 * l + 1]) + "\n";
    }
    return text;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
