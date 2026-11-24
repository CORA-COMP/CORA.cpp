// test_stl_get_variables - the variables of a formula
//
// CORA: specification/@stl/getVariables.m
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


/// The variables of a formula.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl p = x[0] < 5.0, q = x[1] >= 0.0;

    const auto y = StlExpr::variables("y", 2);
    const Stl phi = (x[1] < 1.0) & (y[0] + x[0] > 0.0).finally(StlInterval(0, 2));
    check(phi.get_variables() == std::vector<std::string>({"x1", "x2", "y1"}), b + ": names");
    check(Stl(true).get_variables().empty(), b + ": none");
    // shorter names first, as CORA sorts
    const auto z = StlExpr::variables("z", 10);
    check(((z[9] < 0.0) & (z[1] < 0.0)).get_variables() == std::vector<std::string>({"z2", "z10"}),
          b + ": z2 before z10");
    check(p.next(1).get_variables() == std::vector<std::string>({"x1"}), b + ": inside next");
    {
        ChecksScope off(false);
        check(p.get_variables().size() == 1, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_get_variables");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
