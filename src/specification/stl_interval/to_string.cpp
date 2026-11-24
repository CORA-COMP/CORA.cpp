// to_string - the text of a time interval, as CORA's display
//
// Syntax:   text = I.to_string();   std::cout << I;
// Inputs:   (none)
// Outputs:  text - the interval as "[lb, ub)"
// See also: stl_interval.h
//
// CORA: specification/@stlInterval/display.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl_interval.h"

#include <ostream>
#include <sstream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string StlInterval::to_string() const {
    detail::Entry entry;
    if (empty_) return "{}";
    std::ostringstream os;
    os << (lc_ ? '[' : '(') << lower_ << ", " << upper_ << (rc_ ? ']' : ')');
    return os.str();
}

std::ostream &operator<<(std::ostream &os, const StlInterval &I) { return os << I.to_string(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
