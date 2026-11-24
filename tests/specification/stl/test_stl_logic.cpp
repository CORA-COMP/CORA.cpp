// test_stl_logic - comparisons and the logic operators of STL formulas
//
// CORA: specification/@stl/stl.m
// CORA: specification/@stl/and.m
// CORA: specification/@stl/or.m
// CORA: specification/@stl/not.m
// CORA: specification/@stl/implies.m
// CORA: specification/@stl/lt.m
// CORA: specification/@stl/le.m
// CORA: specification/@stl/gt.m
// CORA: specification/@stl/ge.m
// CORA: specification/@stl/isequal.m
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


/// The sides of the comparisons, the logic nodes, and the equality of formulas.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    // a number on the right stays, on the left swaps the sides, two expressions subtract
    const Stl a = x[0] < 5.0, c = StlExpr(5.0) < x[0], d = x[0] <= x[1];
    check(a.kind() == StlKind::comparison && a.cmp() == StlCmp::lt && a.expr_rhs().value() == 5.0,
          b + ": number right");
    check(c.cmp() == StlCmp::gt && c.expr_lhs().isequal(x[0]) && c.expr_rhs().value() == 5.0,
          b + ": number left");
    check(d.cmp() == StlCmp::le && d.expr_lhs().kind() == StlExprKind::sub &&
              d.expr_rhs().value() == 0.0,
          b + ": two expressions");
    check((x[0] >= 1.0).cmp() == StlCmp::ge && (StlExpr(1.0) >= x[0]).cmp() == StlCmp::le &&
              (StlExpr(1.0) > x[0]).cmp() == StlCmp::lt && (x[0] > 1.0).cmp() == StlCmp::gt,
          b + ": all four");
    check(stl_cmp_dual(StlCmp::lt) == StlCmp::gt && stl_cmp_negate(StlCmp::lt) == StlCmp::ge &&
              stl_cmp_negate(StlCmp::ge) == StlCmp::lt && stl_cmp_dual(StlCmp::ge) == StlCmp::le,
          b + ": dual and negation");
    // the logic nodes build and evaluate nothing; implies is ~a | b
    const Stl p = x[0] < 0.0, q = x[1] < 0.0;
    check((p & q).kind() == StlKind::and_ && (p | q).kind() == StlKind::or_ &&
              (~p).kind() == StlKind::not_,
          b + ": kinds");
    check(p.implies(q).isequal(~p | q) && p.implies(q).kind() == StlKind::or_, b + ": implies");
    check(Stl(true).kind() == StlKind::true_ && Stl(false).kind() == StlKind::false_ &&
              !Stl(true).temporal(),
          b + ": constants");
    check(!(p & q).temporal() && (p & q.next(1)).temporal() && (~p.next(1)).temporal(),
          b + ": temporal");
    // equality: & and | commute, ids are ignored, the windows compare by their bounds
    check((p & q).isequal(q & p) && (p | q).isequal(q | p) && !(p & q).isequal(p | q),
          b + ": commute");
    check(p.with_id(3).isequal(p) && p.with_id(3).id() == 3, b + ": ids");
    check(!p.isequal(q) && !p.isequal(x[0] <= 0.0) && !p.isequal(Stl(true)), b + ": differ");
    check(p.finally(StlInterval(0, 1)).isequal(p.finally(StlInterval(0, 1, false))) &&
              !p.finally(StlInterval(0, 1)).isequal(p.globally(StlInterval(0, 1))),
          b + ": windows");
    check(p.next(1).isequal(p.next(1)) && !p.next(1).isequal(p.next(2)) && (~p).isequal(~p) &&
              Stl(true).isequal(Stl(true)),
          b + ": the rest");
    check(p.until(q, StlInterval(0, 1)).isequal(p.until(q, StlInterval(0, 1))) &&
              !p.until(q, StlInterval(0, 1)).isequal(q.until(p, StlInterval(0, 1))),
          b + ": until");
    check(p == p && p != q, b + ": operators");
    // the sampled semantics of a random non-temporal formula is that of its parts
    Rng rng(4);
    for (int k = 0; k < 20; ++k) {
        const Stl phi = stl_test::random_formula(rng, 3), psi = stl_test::random_formula(rng, 3);
        for (const auto &s : stl_test::states()) {
            check(stl_test::holds(phi & psi, s) ==
                      (stl_test::holds(phi, s) && stl_test::holds(psi, s)),
                  b + ": and");
            check(stl_test::holds(phi.implies(psi), s) ==
                      (!stl_test::holds(phi, s) || stl_test::holds(psi, s)),
                  b + ": implies");
        }
    }
    {
        ChecksScope off(false);
        check((x[0] < 1.0).cmp() == StlCmp::lt, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_logic");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
