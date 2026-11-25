// test_polygon_is_intersecting - Polygon::is_intersecting: whether two polygons meet
//
// CORA: contSet/@polygon/isIntersecting_.m
// PROPERTY: contSet/@polygon/isIntersecting_.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polygon_testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Stated property: random polygons meet iff their intersection or a shared point exists; for
/// pieces that are boxes apart it is false, for overlapping ones true.
void random_pairs() {
    Rng rng(7);
    for (int trial = 0; trial < 20; ++trial) {
        const Polygon A = test::random_polygon(rng, 2), B = test::random_polygon(rng, 2);
        const bool overlap = A.intersect(B).volume() > 0;
        check(!overlap || A.is_intersecting(B), "is_intersecting: an overlap is found");
        const Polygon far = B.plus(vec({100.0, 0.0}));
        check(!A.is_intersecting(far), "is_intersecting: far apart, disjoint");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    random_pairs();
    const Polygon touching = test::boundary({{1, 0}, {2, 0}, {2, 1}, {1, 1}});
    check(test::unit_square().is_intersecting(touching), "is_intersecting: a shared edge meets");
    check(Set(test::unit_square()).get().class_id() == ClassId::polygon &&
              is_intersecting(Set(test::unit_square()), Set(touching)),
          "is_intersecting: the registry");
    check(test::raises([&] { test::unit_square().is_intersecting(touching, 0.5); },
                       Error::not_supported),
          "is_intersecting: a tolerance is not supported");
    {
        ChecksScope off(false);
        check(test::unit_square().is_intersecting(test::l_shape()),
              "is_intersecting: the same with the checks off");
    }
    return test::finish("polygon is_intersecting");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
