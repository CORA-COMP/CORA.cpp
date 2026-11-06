// test_con_zonotope_display - ConZonotope::display: the text that describes the set
//
// CORA: contSet/@conZonotope/display.m
// CORA authors: Dmitry Grebenyuk, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"

#include <sstream>

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const ConZonotope cz(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 3.0}}), Tensor({{1.0, 1.0}}),
                         column({0.5}));
    const std::string text = cz.display();
    check(text.find("constrained zonotope") != std::string::npos, "display: the class");
    check(text.find("dimension: 2") != std::string::npos, "display: the dimension");
    check(text.find("generators (2)") != std::string::npos, "display: the generators");
    check(text.find("constraint matrix A (1 rows)") != std::string::npos,
          "display: the constraints");
    std::ostringstream out;
    out << cz;
    check(out.str() == text, "display: operator<< prints the same text");
    const ConZonotope batch(Tensor::zeros({2, 2, 1}), Tensor::ones({2, 2, 3}));
    check(batch.display().find("batch of shape") != std::string::npos, "display: a batch: shapes");
    // display takes no input; with the checks off the text is the same
    ChecksScope off(false);
    check(cz.display() == text, "display: the same with the checks off");
    return test::finish("con_zonotope display");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
