// test_stl_assign_identifiers - the numbering of the non-temporal parts of a formula
//
// CORA: specification/@stl/assignIdentifiers.m
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


/// The numbering of the non-temporal parts of a formula.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl p = x[0] < 5.0, q = x[1] >= 0.0;

    const StlInterval w = StlInterval(0, 2);
    const Stl r = x[0] > 1.0;
    // p & q is one part, r another; the temporal operators are not numbered
    const Stl phi = (p & q).finally(w) & r.globally(w);
    const auto [res, list] = phi.assign_identifiers();
    check(list.size() == 2 && list[0].isequal(p & q) && list[1].isequal(r), b + ": the list");
    check(list[0].id() == 1 && list[1].id() == 2, b + ": the ids");
    check(res.isequal(phi) && res.id() == 0 && res.lhs().lhs().id() == 1 &&
              res.rhs().lhs().id() == 2,
          b + ": the formula");
    // a non-temporal formula is numbered as a whole, a temporal ~ and until by their operands
    const auto [res2, list2] = (p & q).assign_identifiers();
    check(list2.size() == 1 && res2.id() == 1, b + ": whole");
    const auto [res3, list3] = (~p.until(q, w)).assign_identifiers();
    check(list3.size() == 2 && res3.lhs().lhs().id() == 1 && res3.lhs().rhs().id() == 2,
          b + ": until");
    const auto [res4, list4] = (p.next(1) | (q | r)).assign_identifiers();
    check(list4.size() == 2 && list4[1].isequal(q | r), b + ": or");
    {
        ChecksScope off(false);
        check(phi.assign_identifiers().second.size() == 2, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_assign_identifiers");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
