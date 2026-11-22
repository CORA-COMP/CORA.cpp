// test_interval_is_bounded - interval is_bounded: whether no bound is infinite
//
// CORA: contSet/@interval/isBounded.m
// CORA authors: Mark Wetzlinger, Tobias Ladner
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

void is_bounded_looks_for_infinite_bounds(const std::string &b) {
    const Interval I(vec({-1.0, 0.0}), vec({1.0, 2.0}));
    check(I.is_bounded().all(), b + ": a box is bounded");
    check(!Interval(vec({-kInf, 0.0}), vec({1.0, 2.0})).is_bounded().all(), b + ": -Inf below");
    check(!Interval(vec({-1.0, 0.0}), vec({1.0, kInf})).is_bounded().all(), b + ": Inf above");
    check(!Interval::unbounded(2).is_bounded().all(), b + ": the whole space");
    check(Interval::empty(2).is_bounded().all(), b + ": the empty interval is bounded");
    const Tensor lo = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    check(Interval::matrix(lo, lo + 1.0).is_bounded().all(), b + ": an interval matrix");
    check(!Interval::matrix(lo, lo.full_like({2, 2}, kInf)).is_bounded().all(),
          b + ": an unbounded matrix");
    check(!Interval::stack({I, Interval(vec({0.0, 0.0}), vec({1.0, kInf}))}).is_bounded().all(),
          b + ": a batch is bounded only if every member is");
    check(Interval::stack({I, I}).is_bounded().all(), b + ": a bounded batch");
}

/// is_bounded has no input to refuse; with the checks off the answer is the same.
void checks_and_their_switch() {
    const Interval I(vec({-1.0, 0.0}), vec({1.0, 1.0}));
    ChecksScope off(false);
    check(I.is_bounded().all(), "is_bounded: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(is_bounded_looks_for_infinite_bounds);
    checks_and_their_switch();
    return test::finish("interval is_bounded");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
