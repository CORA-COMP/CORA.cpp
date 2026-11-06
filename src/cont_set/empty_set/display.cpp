// display - the text that describes the empty set, as CORA's emptySet.display
//
// Syntax:   std::string text = E.display();   std::cout << E;
// Inputs:   -
// Outputs:  text - its class and dimension, "n-dimensional empty set"
// See also: Interval::display
//
// CORA: contSet/@emptySet/display.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string EmptySet::display_impl() const {
    return "emptySet\n  " + std::to_string(n_) + "-dimensional empty set\n";
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
