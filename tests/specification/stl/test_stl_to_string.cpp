// test_stl_to_string - the text of a formula
//
// CORA: specification/@stl/disp.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/settings.h"
#include "cora/specification/stl.h"
#include "stl_testing.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {


/// The text of a formula.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl p = x[0] < 5.0, q = x[1] >= 0.0;

    check(p.to_string() == "x1 < 5" && q.to_string() == "x2 >= 0", b + ": comparisons");
    check((2.0 * x[0] + x[1] - 3.5).le(1.0).to_string() == "2*x1 + x2 - 3.5 <= 1", b + ": sums");
    check((x[0] - x[1] >= 0.0).to_string() == "x1 - x2 >= 0", b + ": difference");
    check((p & q).to_string() == "((x1 < 5) & (x2 >= 0))", b + ": and");
    check((p | q.finally(StlInterval(0, 2))).to_string() == "((x1 < 5) | F[0, 2](x2 >= 0))",
          b + ": or");
    check((~p).to_string() == "~(x1 < 5)" && (~p.next(1)).to_string() == "~X[1, 1](x1 < 5)",
          b + ": not");
    check(p.until(q, StlInterval(1, 3)).to_string() == "(x1 < 5 U[1, 3] x2 >= 0)", b + ": until");
    check(Stl(true).to_string() == "true" && (~Stl(false)).to_string() == "~false",
          b + ": constants");
    check(p.globally(StlInterval(0, 2)).to_string() == "G[0, 2](x1 < 5)", b + ": globally");
    {
        ChecksScope off(false);
        check(p.to_string() == "x1 < 5", b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_to_string");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
