// display - the text that describes a level set, as CORA's levelSet.display
//
// The dimension and the comparisons of the constraints. CORA prints the symbolic equations; an
// Expr has no text form, so the constraints themselves are not printed.
//
// Syntax:   std::string text = ls.display();   std::cout << ls;
// Inputs:   -
// Outputs:  text - the class, the dimension and the comparisons
// See also: -
//
// CORA: contSet/@levelSet/display.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/level_set/level_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string LevelSet::display_impl() const {
    std::string text = "levelSet\n- dimension: " + std::to_string(dim()) + "\n- constraints:";
    for (const CompOp op : ops_)
        text += op == CompOp::le ? " f <= 0" : op == CompOp::lt ? " f < 0" : " f == 0";
    return text + "\n";
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
