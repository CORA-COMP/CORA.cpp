// display - the text that describes an interval, as CORA's interval.display
//
// Syntax:   std::string text = I.display();   std::cout << I;
// Inputs:   I - interval (a box, or an interval matrix)
// Outputs:  text - its dimension and bounds, one [inf, sup] row per dimension for a box
// See also: zonotope display, global/format.h

#include "contSet/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string Interval::display() const {
    std::string text = "interval\n- dimension: " + std::to_string(dim()) + "\n";
    const std::vector<int64_t> shape = inf.shape();
    // A box lists its bounds as [inf, sup] per dimension; a matrix shows both bound matrices.
    if (shape.size() != 2 || shape[1] != 1)
        return text + "- infimum:\n" + formatMatrix(inf, "    ") + "- supremum:\n" +
               formatMatrix(sup, "    ");
    const std::vector<double> lo = inf.data(), hi = sup.data();
    text += "- bounds:\n";
    for (std::size_t i = 0; i < lo.size(); ++i)
        text += "    [" + formatNumber(lo[i]) + ", " + formatNumber(hi[i]) + "]\n";
    return text;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
