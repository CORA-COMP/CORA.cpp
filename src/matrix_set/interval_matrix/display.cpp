// display - the text that describes an interval matrix, as CORA's intervalMatrix.display
//
// Syntax:   std::string text = A.display();   std::cout << A;
// Inputs:   -
// Outputs:  text - the class, the dimension (rows and columns) and both bound matrices; CORA
//           prints the interval's display after the dimension. getPrintSetInfo (the abbreviation
//           and property order of that display) has no counterpart: the text is built here
// See also: Interval::display, src/global/format.h
//
// CORA: matrixSet/@intervalMatrix/display.m

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string IntervalMatrix::display() const {
    detail::Entry entry;
    const std::string dimension = std::to_string(rows()) + "  " + std::to_string(cols());
    return "intervalMatrix\n- dimension: " + dimension + "\n- infimum:\n" +
           format_matrix(int_.inf(), "    ") + "- supremum:\n" + format_matrix(int_.sup(), "    ");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
