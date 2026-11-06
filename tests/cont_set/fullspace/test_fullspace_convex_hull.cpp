// test_fullspace_convex_hull - fullspace convex_hull: the hull with any set is the whole space
//
// CORA: contSet/@fullspace/convHull_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/registry.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void convex_hull_is_the_whole_space(const std::string &b) {
    const Fullspace F(2);
    const Interval I(column({0.0, 1.0}), column({2.0, 3.0}));
    check(F.convex_hull(I).dim() == 2 && F.convex_hull(Fullspace(2)).dim() == 2,
          b + ": with a box and with the space");
    check(convex_hull(F, I).is<Fullspace>() && convex_hull(I, F).is<Fullspace>(),
          b + ": through the registry, in either order");
}

/// A set of another dimension is refused; the checks off give the same answer elsewhere.
void checks_and_their_switch() {
    const Fullspace F(2);
    check(test::raises([&] { F.convex_hull(Fullspace(3)); }, Error::dimension_mismatch),
          "convex_hull: another dimension");
    ChecksScope off(false);
    check(F.convex_hull(Fullspace(2)).dim() == 2, "convex_hull: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(convex_hull_is_the_whole_space);
    checks_and_their_switch();
    return test::finish("fullspace convex_hull");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
