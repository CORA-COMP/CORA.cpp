// test_polygon_display - Polygon::display: the text of the pieces
//
// CORA: contSet/@polygon/display.m
// CORA authors: Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polygon_testing.h"

#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void text() {
    const std::string s = test::unit_square().display();
    check(s.find("polygon") != std::string::npos && s.find("pieces: 1") != std::string::npos,
          "display: the class and the number of pieces");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    text();
    {
        ChecksScope off(false);
        check(!test::l_shape().display().empty(), "display: the same with the checks off");
    }
    return test::finish("polygon display");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
