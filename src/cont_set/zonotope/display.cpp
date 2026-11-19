// display - the text that describes a zonotope, as CORA's zonotope.display
//
// Syntax:   std::string text = Z.display();   std::cout << Z;
// Inputs:   Z - zonotope
// Outputs:  text - its dimension, center and generators (a batch: only its shape)
// See also: interval display, src/global/format.h
//
// CORA: contSet/@zonotope/display.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string Zonotope::display_impl() const {
    std::string text = "zonotope\n";
    text += "- dimension: " + std::to_string(dim()) + "\n";
    // a vector zonotope shows CORA's columns, a matrix zonotope its matrices
    const bool vector = elem_rank_ == 1;
    text += "- center:\n" + format_matrix(vector ? c_.expand_dims(-1) : c_, "    ");
    text += "- generators (" + std::to_string(nr_generators()) + "):\n" +
            format_matrix(vector ? G_.matrix_transpose() : G_, "    ");
    return text;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
