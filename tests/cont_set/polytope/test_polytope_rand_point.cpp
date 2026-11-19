// test_polytope_rand_point - Polytope::rand_point: random points of a polytope
//
// CORA: contSet/@polytope/randPoint_.m
// PROPERTY: contSet/@polytope/randPoint_.m
//
// Property: every point is in the set; the extreme points are vertices; hit and run covers the set
//
// CORA authors: Mark Wetzlinger, Adrian Kulmburg
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The points of the three types are in the set, in both forms.
void samples(const std::string &b) {
    Rng rng(4);
    for (const char *type : {"standard", "extreme", "uniform:hitAndRun"}) {
        for (const Polytope &P :
             {test::random_polytope(rng, 3, 5), test::triangle_h(), test::square_v()}) {
            const Tensor p = P.rand_point(20, rng, type);
            check(p.shape() == Shape{20, P.dim()} && P.contains(p, "exact", 1e-7),
                  b + ": inside the set, " + type);
        }
    }
    // the default of the generic entry is "standard"
    const Tensor q = static_cast<const ContSet &>(test::square_h()).rand_point(5, rng);
    check(q.shape() == Shape{5, 2} && test::square_h().contains(q, "exact", 1e-7),
          b + ": the generic entry");
    // the extreme points of a square are corners
    const Tensor e = test::square_v().rand_point(10, rng, "extreme");
    bool corners = true;
    for (const double x : e.data())
        corners = corners && (std::abs(x) < 1e-9 || std::abs(x - 1.0) < 1e-9);
    check(corners, b + ": extreme points of the square are corners");
}

/// R^n gives Gaussian points, the empty set none, an unbounded set points of the set.
void special(const std::string &b) {
    Rng rng(4);
    check(Polytope::inf(3).rand_point(4, rng).shape() == Shape{4, 3}, b + ": R^n");
    check(Polytope::empty(3).rand_point(4, rng).shape() == Shape{0, 3}, b + ": the empty set");
    const Polytope halfspace(Tensor({{1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}}), vec({1.0, 1.0, 0.0}));
    check(halfspace.contains(halfspace.rand_point(10, rng, "extreme"), "exact", 1e-7),
          b + ": unbounded: extreme");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        samples(b);
        special(b);
    });
    Rng rng(4);
    check(
        test::raises([&] { test::square_h().rand_point(0, rng, "standard"); }, Error::wrong_value),
        "rand_point: N >= 1");
    check(
        test::raises([&] { test::square_h().rand_point(3, rng, "gauss"); }, Error::no_specific_alg),
        "rand_point: an unknown type");
    check(test::raises([&] { test::square_h().rand_point(3, rng, "uniform:ballWalk"); },
                       Error::not_supported),
          "rand_point: the ball walk is not ported");
    ChecksScope off(false);
    check(test::square_h().rand_point(2, rng, "standard").shape() == Shape{2, 2},
          "rand_point: the same with the checks off");
    return test::finish("polytope rand_point");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
