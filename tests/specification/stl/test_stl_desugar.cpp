// test_stl_desugar - the formula over until only
//
// CORA: specification/@stl/desugar.m
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


/// The formula over until only.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl p = x[0] < 5.0, q = x[1] >= 0.0;

    const StlInterval w = StlInterval(0, 2);
    const Stl t(true);
    check(p.finally(w).desugar().isequal(t.until(p, w)), b + ": finally");
    check(p.globally(w).desugar().isequal(t.until(~p, w).not_()), b + ": globally");
    check(p.next(1).desugar().isequal(t.until(p, StlInterval(1))), b + ": next");
    check(p.release(q, w).desugar().isequal((~p).until(~q, w).not_()), b + ": release");
    check(p.until(q, w).desugar().isequal(p.until(q, w)) && p.desugar().isequal(p),
          b + ": until and atoms stay");
    // the operands are rewritten, below & | ~ too
    check(((p.finally(w)) & ~q.next(1))
              .desugar()
              .isequal((t.until(p, w)) & ~t.until(q, StlInterval(1))),
          b + ": below the logic");
    check(p.finally(w).desugar().temporal(), b + ": still temporal");
    {
        ChecksScope off(false);
        check(p.finally(w).desugar().kind() == StlKind::until, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_desugar");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
