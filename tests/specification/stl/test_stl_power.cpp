// test_stl_power - the integer power of an expression
//
// CORA: specification/@stl/power.m
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


/// The integer power of an expression.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl p = x[0] < 5.0, q = x[1] >= 0.0;

    const StlExpr y = x[0].power(3);
    check(y.kind() == StlExprKind::pow && y.exponent() == 3 && y.left().isequal(x[0]),
          b + ": power");
    check(x[1].power(-2).exponent() == -2 && x[1].power(0).exponent() == 0, b + ": exponents");
    check(!x[0].power(2).isequal(x[0].power(3)), b + ": the exponent counts");
    check(y.to_string() == "x1^3" && (x[0] + x[1]).power(2).to_string() == "(x1 + x2)^2",
          b + ": display");
    // a non-integer exponent is not supported
    check(test::raises([&] { x[0].power(0.5); }, Error::not_supported), b + ": fractional");
    {
        ChecksScope off(false);
        check(x[0].power(2).exponent() == 2, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_power");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
