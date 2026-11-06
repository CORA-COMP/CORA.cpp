// test_interval_is_intersecting - interval is_intersecting: whether two boxes meet
//
// CORA: contSet/@interval/isIntersecting_.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void is_intersecting_compares_the_bounds(const std::string &b) {
    const Interval I(column({0.0, 0.0}), column({2.0, 2.0}));
    check(I.is_intersecting(Interval(column({1.0, 1.0}), column({3.0, 3.0}))), b + ": overlapping");
    check(I.is_intersecting(Interval(column({-1.0, -1.0}), column({3.0, 3.0}))),
          b + ": the second contains the first");
    check(I.is_intersecting(Interval(column({-1.0, 0.5}), column({1.0, 1.5}))),
          b + ": the second starts first");
    check(I.is_intersecting(Interval(column({2.0, 2.0}), column({3.0, 3.0}))), b + ": touching");
    const Interval apart(column({2.5, 0.0}), column({3.0, 1.0}));
    check(!I.is_intersecting(apart), b + ": disjoint in one coordinate");
    check(!apart.is_intersecting(I), b + ": symmetric");
    check(I.is_intersecting(apart, 0.6) && !I.is_intersecting(apart, 0.2),
          b + ": tol bridges a gap");
    check(I.is_intersecting(Interval(column({2.0 + 1e-9, 0.0}), column({3.0, 1.0})), 1e-8),
          b + ": a gap of 1e-9 within tol 1e-8 (CORA's default)");
    check(!I.is_intersecting(Interval(column({2.0 + 1e-9, 0.0}), column({3.0, 1.0})), 0.0),
          b + ": tol 0 is strict");
    // an empty interval meets nothing
    check(!I.is_intersecting(Interval::empty(2)) && !Interval::empty(2).is_intersecting(I),
          b + ": an empty interval meets nothing");
}

void is_intersecting_with_points_and_batches(const std::string &b) {
    const Interval I(column({0.0, 0.0}), column({2.0, 2.0}));
    check(I.is_intersecting(column({1.0, 2.0})) && !I.is_intersecting(column({1.0, 2.5})),
          b + ": a point is in the box or not");
    check(I.is_intersecting(column({1.0, 2.5}), 0.6), b + ": a point within tol");
    const Interval far(column({5.0, 5.0}), column({6.0, 6.0}));
    check(Interval::stack({I, I}).is_intersecting(Interval::stack({I, I})) &&
              !Interval::stack({I, I}).is_intersecting(Interval::stack({I, far})),
          b + ": a batch meets only if every member does");
    check(is_intersecting(I, far) == false && is_intersecting(I, I), b + ": the registry pair");
}

/// CORA tests the first max(rows, columns) entries of an interval matrix; every entry counts here.
void is_intersecting_looks_at_every_matrix_entry(const std::string &b) {
    const Interval A(Tensor::from_data({0, 0, 0, 0}, {2, 2}),
                     Tensor::from_data({1, 1, 1, 1}, {2, 2}));
    const Interval B(Tensor::from_data({0, 5, 0, 5}, {2, 2}),
                     Tensor::from_data({1, 6, 1, 6}, {2, 2}));
    check(!A.is_intersecting(B) && A.is_intersecting(A), b + ": the second column is disjoint");
}

/// A wrong dimension, a negative tol and a point of the wrong shape are refused.
void checks_and_their_switch() {
    const Interval I(column({0.0, 0.0}), column({2.0, 2.0}));
    check(test::raises([&] { I.is_intersecting(Interval(column({0.0}), column({1.0}))); },
                       Error::dimension_mismatch),
          "is_intersecting: dimensions 2 and 1");
    check(test::raises([&] { I.is_intersecting(I, -1.0); }, Error::wrong_value),
          "is_intersecting: a negative tol");
    check(test::raises([&] { I.is_intersecting(column({1.0})); }, Error::dimension_mismatch),
          "is_intersecting: a point of one coordinate");
    ChecksScope off(false);
    check(I.is_intersecting(I), "is_intersecting: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        is_intersecting_compares_the_bounds(b);
        is_intersecting_with_points_and_batches(b);
        is_intersecting_looks_at_every_matrix_entry(b);
    });
    checks_and_their_switch();
    return test::finish("interval is_intersecting");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
