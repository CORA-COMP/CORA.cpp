// test_fullspace_vertices - fullspace vertices: the corners of the unbounded box
//
// CORA: contSet/@fullspace/vertices_.m
// PROPERTY: contSet/@fullspace/vertices_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

#include <cmath>
#include <set>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void the_corners_are_at_infinity(const std::string &b) {
    const std::vector<Polygon> V = Fullspace(2).vertices();
    check(V.size() == 1 && V[0].size() == 4, b + ": one polygon of four corners");
    std::set<std::pair<double, double>> signs;
    for (const Point &p : V[0]) {
        check(std::isinf(p[0]) && std::isinf(p[1]), b + ": both coordinates infinite");
        signs.insert({p[0], p[1]});
    }
    check(signs.size() == 4, b + ": every sign combination occurs once");
}

/// The corners are those of the interval hull, and the space contains the vertices of random
/// zonotopes of a fixed seed (the vertices of a subset lie in the set).
void the_corners_are_those_of_the_box(const std::string &b) {
    check(Fullspace(2).vertices() == Fullspace(2).interval().vertices(), b + ": as interval(fs)");
    Rng rng(13);
    for (int trial = 0; trial < 10; ++trial) {
        const Zonotope Z = Zonotope::generate_random(rng, 2, std::nullopt, 4);
        for (const Polygon &polygon : Z.vertices())
            for (const Point &p : polygon)
                check(Fullspace(2).contains(
                          Interval(test::column({p[0], p[1]}), test::column({p[0], p[1]}))),
                      b + ": a vertex of a zonotope is in the whole space");
    }
}

/// A set of three dimensions has no polygon; with the checks off the result is the same.
void checks_and_their_switch() {
    check(test::raises([] { Fullspace(3).vertices(); }, Error::not_supported), "vertices: R^3");
    check(test::raises([] { Fullspace(0).vertices(); }, Error::not_supported), "vertices: R^0");
    ChecksScope off(false);
    check(Fullspace(2).vertices()[0].size() == 4, "vertices: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(the_corners_are_at_infinity);
    test::for_each_backend(the_corners_are_those_of_the_box);
    checks_and_their_switch();
    return test::finish("fullspace vertices");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
