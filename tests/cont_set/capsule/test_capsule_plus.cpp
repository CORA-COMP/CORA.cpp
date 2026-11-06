// test_capsule_plus - Capsule plus: the sum with a capsule, a vector, a box and a zonotope
//
// CORA: contSet/@capsule/plus.m
// PROPERTY: contSet/@capsule/plus.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"
#include "cora/cont_set/zonotope/zonotope.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: the longer generator stays, the shorter one joins the radius.
void plus_known(const std::string &b) {
    const Capsule A(column({0.0, 0.0}), column({3.0, 4.0}), 1.0),
        B(column({1.0, 1.0}), column({1.0, 0.0}), 0.5);
    for (const Capsule &S : {A.plus(B), B.plus(A)}) {
        check(close(S.c(), {1, 1}, 0) && close(S.generator(), {3, 4}, 0),
              b + ": c adds, the longer g stays");
        check(close(S.r(), {2.5}, 1e-14), b + ": r = 1 + 0.5 + |(1, 0)|");
    }
    // a vector shifts, a box shifts and widens by the radius of the box, a zonotope by its box
    check(close(A.plus(column({1.0, 2.0})).c(), {1, 2}, 0), b + ": a vector");
    const Interval I(column({0.0, 0.0}), column({2.0, 4.0}));
    const Capsule with_box = A.plus(I);
    check(close(with_box.c(), {1, 2}, 0) && close(with_box.r(), {1.0 + std::sqrt(5.0)}, 1e-14),
          b + ": a box");
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 2.0}}));
    check(close(A.plus(Z).r(), with_box.r(), 0), b + ": a zonotope is its box");
    check(A.plus(Capsule::empty(2)).is_empty() && Capsule::empty(2).plus(A).is_empty(),
          b + ": empty");
    check(plus(A, I).is<Capsule>() && plus(Z, A).is<Capsule>(), b + ": through the registry");
}

/// Stated property: the sum of a point of each summand lies in the enclosure, whatever the lengths.
void plus_property(const std::string &b) {
    Rng rng(47);
    for (int k = 0; k < 6; ++k) {
        const Capsule A = test::random_capsule(rng, 3), B = test::random_capsule(rng, 3);
        const Tensor sum = test::sample_points(A, rng, 25) + test::sample_points(B, rng, 25);
        check(A.plus(B).contains(sum), b + ": the sums are inside");
        check(B.plus(A).contains(sum), b + ": in the other order too");
    }
}

/// A capsule of another dimension is refused; with the checks off a valid call is the same.
void checks_and_their_switch() {
    const Capsule A(column({0.0, 0.0}), column({3.0, 4.0}), 1.0);
    check(test::raises([&] { A.plus(Capsule(column({1.0}))); }, Error::dimension_mismatch),
          "plus: dimension");
    check(test::raises([&] { A.plus(column({1.0, 2.0, 3.0})); }, Error::dimension_mismatch),
          "plus: vector");
    const Tensor on = A.plus(A).r();
    ChecksScope off(false);
    check(close(A.plus(A).r(), on, 0), "plus: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        plus_known(b);
        plus_property(b);
    });
    checks_and_their_switch();
    return test::finish("capsule plus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
