// display - the text that describes an ellipsoid, as CORA's ellipsoid.display
//
// Syntax:   std::string text = E.display();   std::cout << E;
// Inputs:   E - ellipsoid
// Outputs:  text - its dimension, center and shape matrix (a batch: only the shapes)
// See also: Zonotope display, src/global/format.h
//
// CORA: contSet/@ellipsoid/display.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string Ellipsoid::display_impl() const {
    std::string text = "ellipsoid\n";
    text += "- dimension: " + std::to_string(dim()) + "\n";
    text += "- center:\n" + format_matrix(q_.expand_dims(-1), "    ");
    return text + "- shape matrix Q:\n" + format_matrix(Q_, "    ");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
