// test_polytope_display - Polytope::display: the text that describes the set
//
// CORA: contSet/@polytope/display.m

#include "polytope_testing.h"

#include <sstream>

using namespace cora;
using test::check;
using test::column;

int main() {
    const Polytope P(Tensor({{1.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    const std::string text = P.display();
    check(text.find("polytope") != std::string::npos, "display: the class");
    check(text.find("dimension: 2") != std::string::npos, "display: the dimension");
    check(text.find("inequality constraints A x <= b (2 rows)") != std::string::npos,
          "display: the inequalities");
    check(text.find("equality constraints Ae x = be (0 rows)") != std::string::npos,
          "display: the equalities");
    check(text.find("vertices: (not computed)") != std::string::npos, "display: no vertices");
    const Polytope V(Tensor({{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}));
    check(V.display().find("vertices (3)") != std::string::npos, "display: the vertices");
    check(V.display().find("constraints: (not computed)") != std::string::npos,
          "display: no constraints");
    std::ostringstream out;
    out << P;
    check(out.str() == text, "display: operator<< prints the same text");
    const Polytope batch(Tensor::ones({2, 2, 2}), Tensor::ones({2, 2, 1}));
    check(batch.display().find("batch of shape") != std::string::npos, "display: a batch: shapes");
    // display takes no input; with the checks off the text is the same
    ChecksScope off(false);
    check(P.display() == text, "display: the same with the checks off");
    return test::finish("polytope display");
}
