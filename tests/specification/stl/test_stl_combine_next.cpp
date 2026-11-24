// test_stl_combine_next - the merging of nested next operators
//
// CORA: specification/@stl/combineNext.m
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


/// The merging of nested next operators.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl p = x[0] < 5.0, q = x[1] >= 0.0;

    const StlInterval w = StlInterval(0, 2);
    check(p.next(1).next(2).combine_next().isequal(p.next(3)), b + ": next of next");
    check(p.next(1).next(2).next(0.5).combine_next().isequal(p.next(3.5)), b + ": three");
    // next distributes over a temporal & | and over ~, and stays at a non-temporal one
    check((p.finally(w) & q).next(1).combine_next().isequal(p.finally(w).next(1) & q.next(1)),
          b + ": and");
    check((p.finally(w) | q).next(1).combine_next().isequal(p.finally(w).next(1) | q.next(1)),
          b + ": or");
    check((p & q).next(1).combine_next().isequal((p & q).next(1)), b + ": non-temporal");
    check((~p.next(1)).next(2).combine_next().isequal(~p.next(3)), b + ": negation");
    check(p.next(1).next(2).finally(w).combine_next().isequal(p.next(1).next(2).finally(w)),
          b + ": under finally nothing is merged");
    check((p.next(1).next(1) & q).combine_next().isequal(p.next(2) & q), b + ": inside &");
    check(p.combine_next().isequal(p), b + ": atoms");
    {
        ChecksScope off(false);
        check(p.next(1).next(2).combine_next().isequal(p.next(3)), b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_combine_next");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
