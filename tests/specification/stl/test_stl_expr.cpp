// test_stl_expr - the arithmetic of STL expressions
//
// CORA: specification/@stl/stl.m
// CORA: specification/@stl/plus.m
// CORA: specification/@stl/minus.m
// CORA: specification/@stl/times.m
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


/// The variables, the operand arrangement of the operators, display and equality.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 3);
    check(x.size() == 3 && x[1].name() == "x2" && x[2].index() == 2, b + ": variables");
    check(x[0].kind() == StlExprKind::variable && StlExpr(2.5).value() == 2.5, b + ": kinds");
    // a numeric left operand of + moves to the right, of * it stays first, of - it stays
    const StlExpr s = StlExpr(2.0) + x[0];
    check(s.kind() == StlExprKind::add && s.left().isequal(x[0]) && s.right().value() == 2.0,
          b + ": plus");
    const StlExpr t = x[0] * 3.0, u = x[0] * x[1];
    check(t.left().value() == 3.0 && t.right().isequal(x[0]), b + ": number first");
    check(u.left().isequal(x[1]) && u.right().isequal(x[0]), b + ": the right factor first");
    const StlExpr d = StlExpr(1.0) - x[0];
    check(d.kind() == StlExprKind::sub && d.left().value() == 1.0, b + ": minus");
    // equality: sums and products commute, differences do not
    check((x[0] + x[1]).isequal(x[1] + x[0]) && (x[0] * x[1]).isequal(x[1] * x[0]),
          b + ": commute");
    check(!(x[0] - x[1]).isequal(x[1] - x[0]) && !x[0].isequal(x[1]) && !x[0].isequal(2.0),
          b + ": differ");
    check(x[0].isequal(StlExpr::variable("x1", 0)), b + ": by name");
    // the names, ordered as getVariables does, and the linearity
    check(((x[2] + x[0]) * 2.0 - x[0]).variable_names() == std::vector<std::string>({"x1", "x3"}),
          b + ": names");
    check((x[0] * 2.0 + 1.0 - x[1]).is_linear() && !(x[0] * x[1]).is_linear(), b + ": linear");
    check(StlExpr(2.0).to_string() == "2" && StlExpr(0.25).to_string() == "0.25", b + ": numbers");
    // an empty name or no variable is refused
    check(test::raises([] { StlExpr::variables("", 2); }, Error::wrong_input_in_constructor),
          b + ": empty name");
    check(test::raises([] { StlExpr::variables("x", 0); }, Error::wrong_input_in_constructor),
          b + ": no variable");
    {
        ChecksScope off(false);
        check(StlExpr::variables("y", 2).size() == 2, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_expr");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
