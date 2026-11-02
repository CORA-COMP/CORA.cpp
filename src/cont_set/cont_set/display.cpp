// display - the text that describes a set, as CORA's contSet.display
//
// Syntax:   text = S.display();   std::cout << S;
// Inputs:   S - set
// Outputs:  text - its class, dimension and parameters (a batch: only their shape)
// See also: Zonotope::display_impl, Interval::display_impl
//
// CORA: contSet/@contSet/display.m

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string ContSet::display() const {
    detail::Entry entry;
    return display_impl();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
