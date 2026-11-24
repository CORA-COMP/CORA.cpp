// test_stl_finally - the finally operator over a time window
//
// CORA: specification/@stl/finally.m
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


/// The finally operator over a time window.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl p = x[0] < 5.0, q = x[1] >= 0.0;

    const Stl phi = p.finally(StlInterval(0, 2));
    check(phi.kind() == StlKind::finally && phi.from() == 0 && phi.to() == 2 && phi.temporal(),
          b + ": finally");
    check(phi.lhs().isequal(p), b + ": the operand");
    check(q.finally(StlInterval(1)).to() == 1, b + ": a point window");
    // an empty window is refused
    check(test::raises([&] { p.finally(StlInterval()); }, Error::wrong_value),
          b + ": empty window");
    {
        ChecksScope off(false);
        check(p.finally(StlInterval(0, 2)).to() == 2, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_finally");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
