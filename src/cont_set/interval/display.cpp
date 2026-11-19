// display - the text that describes an interval, as CORA's interval.display
//
// Syntax:   std::string text = I.display();   std::cout << I;
// Inputs:   I - interval (a box, or an interval matrix)
// Outputs:  text - its dimension and bounds, one [inf, sup] row per dimension for a box
// See also: zonotope display, src/global/format.h
//
// CORA: contSet/@interval/display.m
// CORA: matrixSet/@intervalMatrix/display.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string Interval::display_impl() const {
    const std::vector<int64_t> shape = inf_.shape();
    // an interval matrix shows both bound matrices, as CORA's intervalMatrix display
    if (elem_rank_ == 2 && shape.size() == 2)
        return "intervalMatrix\n- dimension: " + std::to_string(shape[0]) + "  " +
               std::to_string(shape[1]) + "\n- infimum:\n" + format_matrix(inf_, "    ") +
               "- supremum:\n" + format_matrix(sup_, "    ");
    std::string text = "interval\n- dimension: " + std::to_string(dim()) + "\n";
    if (is_empty()) return text + "- bounds:\n"; // no bounds to list
    // a box lists its bounds as [inf, sup] per dimension, a batch both bound tensors
    if (shape.size() != 1)
        return text + "- infimum:\n" + format_matrix(inf_, "    ") + "- supremum:\n" +
               format_matrix(sup_, "    ");
    const std::vector<double> lo = inf_.data(), hi = sup_.data();
    text += "- bounds:\n";
    for (std::size_t i = 0; i < lo.size(); ++i)
        text += "    [" + format_number(lo[i]) + ", " + format_number(hi[i]) + "]\n";
    return text;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
