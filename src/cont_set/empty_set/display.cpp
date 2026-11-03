// display - the text that describes the empty set, as CORA's emptySet.display
//
// Syntax:   std::string text = E.display();   std::cout << E;
// Inputs:   -
// Outputs:  text - its class and dimension, "n-dimensional empty set"
// See also: Interval::display
//
// CORA: contSet/@emptySet/display.m

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string EmptySet::display_impl() const {
    return "emptySet\n  " + std::to_string(n_) + "-dimensional empty set\n";
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
