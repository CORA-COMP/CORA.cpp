// test_stl_until - the until operator of two formulas over a time window
//
// CORA: specification/@stl/until.m
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


/// The until operator of two formulas over a time window.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl p = x[0] < 5.0, q = x[1] >= 0.0;

    const Stl phi = p.until(q, StlInterval(1, 3));
    check(phi.kind() == StlKind::until && phi.from() == 1 && phi.to() == 3 && phi.temporal(),
          b + ": until");
    check(phi.lhs().isequal(p) && phi.rhs().isequal(q), b + ": the operands");
    check(!p.until(q, StlInterval(0, 1)).isequal(q.until(p, StlInterval(0, 1))),
          b + ": not symmetric");
    // an empty window is refused
    check(test::raises([&] { p.until(q, StlInterval()); }, Error::wrong_value),
          b + ": empty window");
    {
        ChecksScope off(false);
        check(p.until(q, StlInterval(0, 1)).to() == 1, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_until");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
