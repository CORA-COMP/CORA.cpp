// test_con_zonotope_contains - ConZonotope::contains: points, zonotopes and boxes in the set
//
// CORA: contSet/@conZonotope/contains_.m
// PROPERTY: contSet/@conZonotope/contains_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger, Matthias Althoff, Adrian Kulmburg, Ivan Brkan
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"
#include "cora/cont_set/registry.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: the segment x = xi, xi1 + xi2 = 1 holds its points and nothing else.
void contains_known(const std::string &b) {
    const ConZonotope segment(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                              Tensor({{1.0, 1.0}}), column({1.0}));
    check(segment.contains(column({0.5, 0.5})) && segment.contains(column({0.0, 1.0})),
          b + ": points of the segment");
    check(!segment.contains(column({0.5, 0.6})) && !segment.contains(column({1.5, -0.5})),
          b + ": a point off the line, a point on the line beyond the box");
    check(segment.contains(Tensor({{0.5, 1.0}, {0.5, 0.0}})), b + ": several points: all in");
    check(!segment.contains(Tensor({{0.5, 1.0}, {0.5, 1.0}})), b + ": several points: one out");
    check(segment.contains(column({0.5, 0.5 + 1e-9}), "exact", 1e-6),
          b + ": a tolerance admits a point off the line");
    check(segment.contains(Zonotope(column({0.5, 0.5}), Tensor({{0.25}, {-0.25}}))),
          b + ": a zonotope on it");
    check(!segment.contains(Zonotope(column({0.5, 0.5}), Tensor({{0.25}, {0.25}}))),
          b + ": a zonotope across it");
    check(segment.contains(Interval(column({0.5, 0.5}), column({0.5, 0.5}))), b + ": a point box");
    check(!segment.contains(Interval(column({0.0, 0.0}), column({1.0, 1.0}))),
          b + ": a box is not in a segment");
    // the box [-1, 1]^2 as a constrained zonotope with a trivial constraint
    const ConZonotope square(column({0.0, 0.0}), Tensor({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}),
                             Tensor({{0.0, 0.0, 1.0}}), column({0.0}));
    check(square.contains(Interval(column({-0.5, -0.5}), column({0.5, 1.0}))),
          b + ": a box in a square");
    check(!square.contains(Interval(column({-0.5, -0.5}), column({0.5, 1.5}))),
          b + ": a box sticking out");
    check(square.contains(Zonotope(column({0.0, 0.0}), Tensor({{0.5, 0.5}, {0.5, -0.5}}))),
          b + ": a zonotope in a square");
    check(square.contains(column({1.0 + 1e-9, 0.0}), "exact", 1e-6), b + ": a tolerance admits it");
    check(!square.contains(column({1.001, 0.0})), b + ": and the default tolerance does not");
    // a set that is a point: the points within tol
    const ConZonotope point(column({2.0, 3.0}), Tensor({{0.0}, {0.0}}));
    check(point.contains(column({2.0, 3.0 + 1e-10}), "exact", 1e-8) &&
              !point.contains(column({2.0, 3.1})),
          b + ": a point holds the points within tol");
    // nothing is in the empty set but no point
    const ConZonotope none = ConZonotope::empty(2);
    check(!none.contains(column({0.0, 0.0})) && none.contains(Tensor::zeros({2, 0})),
          b + ": empty");
    const Set via_set(segment);
    check(contains(via_set, Set(Interval(column({0.5, 0.5}), column({0.5, 0.5})))),
          b + ": through the registry");
}

/// Stated property: the sampled points of a set are in it; the points of a bigger set mostly are
/// not.
void contains_property(const std::string &b) {
    cora::Rng rng(32);
    for (int k = 0; k < 4; ++k) {
        const ConZonotope A = test::random_con_zonotope(rng, 3, 5, 2);
        check(A.contains(test::sample_points(A, rng, 40), "exact", 1e-7),
              b + ": the sampled points");
        check(!A.contains(A.interval().sup() + Tensor::ones({3, 1})), b + ": beyond the box");
        // a small box around the center, and its corners, are in a full-dimensional set
        const Tensor c = A.center();
        check(A.contains(Interval(c - 1e-6, c + 1e-6)), b + ": a tiny box around the center");
    }
}

/// A batch, a wrong shape, an unknown or unported method are refused; the same valid call with the
/// checks off.
void checks_and_their_switch() {
    const ConZonotope A(column({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { A.contains(Tensor({{1.0, 2.0}, {1.0, 2.0}})); },
                       Error::dimension_mismatch),
          "contains: the points are (n, N)");
    check(test::raises([&] { A.contains(Zonotope(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}))); },
                       Error::dimension_mismatch),
          "contains: another dimension");
    check(test::raises([&] { A.contains(column({0.0}), "nothing"); }, Error::no_specific_alg),
          "contains: an unknown method");
    for (const char *method : {"approx", "approx:st", "exact:polymax", "sampling", "sampling:dual"})
        check(test::raises([&] { A.contains(column({0.0}), method); }, Error::not_supported),
              std::string("contains: ") + method + " is not ported");
    check(test::raises([&] { A.contains(column({0.0}), "exact", -1.0); }, Error::wrong_value),
          "contains: tol < 0");
    check(test::raises(
              [&] {
                  // a batch of sets is not supported
                  ConZonotope(Tensor::zeros({2, 1, 1}), Tensor::ones({2, 1, 1}))
                      .contains(column({0.0}));
              },
              Error::not_supported),
          "contains: a batch");
    check(test::raises([&] { A.contains(Zonotope(column({0.0}), Tensor::ones({1, 13}))); },
                       Error::not_supported),
          "contains: too many corners");
    check(A.contains(column({0.5})), "contains: the same call with the checks on");
    ChecksScope off(false);
    check(A.contains(column({0.5})), "contains: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        contains_known(b);
        contains_property(b);
    });
    checks_and_their_switch();
    return test::finish("con_zonotope contains");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
