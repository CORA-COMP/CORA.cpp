// test_con_zonotope_convex_hull - ConZonotope::convex_hull: the exact hull with other sets
//
// CORA: contSet/@conZonotope/convHull_.m
// PROPERTY: contSet/@conZonotope/convHull_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"
#include "cora/cont_set/registry.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: the hull of [0, 1] and [2, 3] is [0, 3], by both methods; a point, a box, a
/// zonotope.
void convex_hull_known(const std::string &b) {
    const ConZonotope A(column({0.5}), Tensor({{0.5}}));
    const ConZonotope B(column({2.5}), Tensor({{0.5}}));
    for (const char *method : {"exact", "exact:Raghuraman", "exact:Kochdumper"}) {
        const ConZonotope H = A.convex_hull(B, method);
        check(close(H.interval().inf(), {0}, 1e-9) && close(H.interval().sup(), {3}, 1e-9),
              b + ": the hull [0, 3] (" + method + ")");
    }
    const ConZonotope R = A.convex_hull(B);
    check(close(R.c(), {1.5}, 0) && close(R.generators(), {0.5, 0.5, -1, 0, 0, 0, 0}, 0),
          b + ": the center is the mean, the generators of both, (c-c2)/2, and the slacks");
    check(R.A().shape() == Shape{4, 7} && R.b().shape() == Shape{4, 1},
          b + ": 2 (m1 + m2) slack rows");
    check(close(R.b(), {-0.5, -0.5, -0.5, -0.5}, 0), b + ": the offsets -1/2");
    check(close(A.convex_hull(column({5.0})).interval().sup(), {5}, 1e-9), b + ": a point");
    check(
        close(A.convex_hull(Interval(column({-2.0}), column({-1.0}))).interval().inf(), {-2}, 1e-9),
        b + ": a box");
    check(
        close(A.convex_hull(Zonotope(column({3.0}), Tensor({{1.0}}))).interval().sup(), {4}, 1e-9),
        b + ": a zonotope");
    // constraints of both sets are kept: the segment x1 + x2 = 1 and the origin give a triangle
    const ConZonotope segment(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                              Tensor({{1.0, 1.0}}), column({1.0}));
    const ConZonotope T = segment.convex_hull(column({0.0, 0.0}));
    check(close(T.interval().inf(), {0, 0}, 1e-9) && close(T.interval().sup(), {1, 1}, 1e-9),
          b + ": the triangle (0,0), (0,1), (1,0)");
    check(T.contains(column({0.3, 0.3})) && !T.contains(column({0.7, 0.7})),
          b + ": inside and outside");
    const Set via_set = convex_hull(A, Set(B));
    check(via_set.is<ConZonotope>(), b + ": through the registry");
}

/// The hull of a list contains this set and every member (CORA's list form leaves the first out).
void convex_hull_of_a_list(const std::string &b) {
    const ConZonotope A(column({0.0}), Tensor({{0.5}})), B(column({3.0}), Tensor({{0.5}})),
        C(column({-4.0}), Tensor({{0.5}}));
    const ConZonotope H = A.convex_hull(std::vector<ConZonotope>{B, C});
    check(close(H.interval().inf(), {-4.5}, 1e-9) && close(H.interval().sup(), {3.5}, 1e-9),
          b + ": the list: all three sets, the first included");
    check(A.convex_hull(std::vector<ConZonotope>{B}).contains(column({1.5})),
          b + ": one other: the segment");
}

/// Stated property: both methods give the same set, and the hull holds the points of both sets.
void convex_hull_property(const std::string &b) {
    cora::Rng rng(29);
    for (int k = 0; k < 3; ++k) {
        const ConZonotope A = test::random_con_zonotope(rng, 2, 3, 1),
                          B = test::random_con_zonotope(rng, 2, 3, 1);
        const ConZonotope R = A.convex_hull(B), K = A.convex_hull(B, "exact:Kochdumper");
        const Tensor pa = test::sample_points(A, rng, 15), pb = test::sample_points(B, rng, 15);
        check(R.contains(pa, "exact", 1e-7) && R.contains(pb, "exact", 1e-7),
              b + ": Raghuraman holds both");
        check(K.contains(pa, "exact", 1e-7) && K.contains(pb, "exact", 1e-7),
              b + ": Kochdumper holds both");
        // a midpoint of two points is in the hull, and the two methods have the same box
        check(R.contains((pa + pb) * 0.5, "exact", 1e-7), b + ": midpoints are in the hull");
        check(close(R.interval().inf(), K.interval().inf(), 1e-7) &&
                  close(R.interval().sup(), K.interval().sup(), 1e-7),
              b + ": the same box by both methods");
    }
}

/// Wrong operands and methods are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const ConZonotope A(column({0.5}), Tensor({{0.5}}));
    check(test::raises(
              [&] { A.convex_hull(ConZonotope(column({1.0, 2.0}), Tensor({{1.0}, {1.0}}))); },
              Error::dimension_mismatch),
          "convex_hull: different dimensions");
    check(test::raises([&] { A.convex_hull(A, "outer"); }, Error::wrong_value),
          "convex_hull: a method");
    check(test::raises([&] { A.convex_hull(Tensor({{1.0, 2.0}})); }, Error::dimension_mismatch),
          "convex_hull: a point is a column");
    check(test::raises(
              [&] {
                  A.convex_hull(std::vector<ConZonotope>{
                      ConZonotope(column({1.0, 2.0}), Tensor({{1.0}, {1.0}}))});
              },
              Error::dimension_mismatch),
          "convex_hull: a list of other dimensions");
    const Tensor on = A.convex_hull(A).A();
    ChecksScope off(false);
    check(close(A.convex_hull(A).A(), on, 0), "convex_hull: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        convex_hull_known(b);
        convex_hull_of_a_list(b);
        convex_hull_property(b);
    });
    checks_and_their_switch();
    return test::finish("con_zonotope convex_hull");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
