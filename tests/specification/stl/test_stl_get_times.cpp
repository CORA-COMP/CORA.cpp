// test_stl_get_times - the times at which the identified parts are evaluated
//
// CORA: specification/@stl/getTimes.m
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


/// The times at which the identified parts are evaluated.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl p = x[0] < 5.0, q = x[1] >= 0.0;

    const Stl r = x[0] > 1.0;
    // F[0,2] (p & q) & X[3] r: part 1 over the window [0, 2], part 2 at the time 3
    const auto [phi, list] = ((p & q).finally(StlInterval(0, 2)) & r.next(3)).assign_identifiers();
    const StlTimes t = phi.get_times();
    check(t.time_int.size() == 2 && t.time_point.size() == 2, b + ": equal lengths");
    check(t.time_int[0].size() == 1 && t.time_int[0][0] == StlInterval(0, 2) &&
              t.time_int[1].empty(),
          b + ": window");
    check(t.time_point[0].empty() && t.time_point[1] == std::vector<double>({3.0}), b + ": point");
    // windows add up under nested operators, a next shifts a window
    const auto [psi, l2] =
        p.next(1).globally(StlInterval(1, 2)).finally(StlInterval(0, 4)).assign_identifiers();
    check(psi.get_times().time_int[0][0] == StlInterval(2, 7), b + ": nested");
    const auto [chi, l3] = p.next(1).finally(StlInterval(1, 2)).assign_identifiers();
    check(chi.get_times().time_int[0][0] == StlInterval(2, 3), b + ": shifted window");
    // an until looks at both operands over its window; a formula without ids has no times
    const auto [ups, l4] = p.until(q, StlInterval(1, 2)).assign_identifiers();
    const StlTimes u = ups.get_times();
    check(u.time_int.size() == 2 && u.time_int[1][0] == StlInterval(1, 2), b + ": until");
    check((p & q).get_times().time_int.empty() &&
              p.finally(StlInterval(0, 2)).get_times().time_point.empty(),
          b + ": no ids");
    // a negated temporal formula is read through (CORA stops there)
    check((~p.finally(StlInterval(0, 2))).assign_identifiers().first.get_times().time_int.size() ==
              1,
          b + ": ~");
    {
        ChecksScope off(false);
        check(phi.get_times().time_int.size() == 2, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_get_times");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
