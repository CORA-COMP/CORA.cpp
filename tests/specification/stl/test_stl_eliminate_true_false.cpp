// test_stl_eliminate_true_false - the formula without true and false below the operators
//
// CORA: specification/@stl/eliminateTrueFalse.m
// PROPERTY: specification/@stl/eliminateTrueFalse.m
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


/// The formula without true and false below the operators.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl p = x[0] < 5.0, q = x[1] >= 0.0;

    const StlInterval w = StlInterval(0, 2);
    const Stl t(true), f(false);
    // true and false in & and |
    check((p & t).eliminate_true_false().isequal(p) && (t & p).eliminate_true_false().isequal(p),
          b + ": true in &");
    check((p & f).eliminate_true_false().kind() == StlKind::false_, b + ": false in &");
    check((p | f).eliminate_true_false().isequal(p) && (f | p).eliminate_true_false().isequal(p),
          b + ": false in |");
    check((p | t).eliminate_true_false().kind() == StlKind::true_, b + ": true in |");
    check((~t).eliminate_true_false().kind() == StlKind::false_ &&
              (~f).eliminate_true_false().kind() == StlKind::true_,
          b + ": negated constants");
    // constants under the temporal operators
    check(f.next(1).eliminate_true_false().kind() == StlKind::false_ &&
              t.finally(w).eliminate_true_false().kind() == StlKind::true_ &&
              f.globally(w).eliminate_true_false().kind() == StlKind::false_,
          b + ": temporal constants");
    check(t.until(p, w).eliminate_true_false().isequal(p.finally(w)), b + ": true U p is F p");
    check(f.release(p, w).eliminate_true_false().isequal(p.globally(w)), b + ": false R p is G p");
    // a constant operand of until and release decides them
    check(p.until(f, w).eliminate_true_false().kind() == StlKind::false_, b + ": until false");
    check(p.release(t, w).eliminate_true_false().kind() == StlKind::true_ &&
              t.release(p, w).eliminate_true_false().kind() == StlKind::true_,
          b + ": release true");
    // CORA rewrites phi U true to true; false U[0,2] true is false, so the node stays
    check(p.until(t, w).eliminate_true_false().isequal(p.until(t, w)), b + ": until true stays");
    check(p.next(1).eliminate_true_false().isequal(p.next(1)) &&
              p.finally(w).eliminate_true_false().isequal(p.finally(w)),
          b + ": nothing to do");
    // an identified part stays as it is
    check((t & p).with_id(2).eliminate_true_false().isequal(t & p), b + ": identified");
    // random formulas keep their meaning and lose their constants
    Rng rng(4);
    for (int k = 0; k < 40; ++k) {
        const Stl phi = stl_test::random_formula(rng, 4);
        check(stl_test::equivalent(phi, phi.eliminate_true_false()), b + ": equivalent");
    }
    {
        ChecksScope off(false);
        check((p & t).eliminate_true_false().isequal(p), b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_eliminate_true_false");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
