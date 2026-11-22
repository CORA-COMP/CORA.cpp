// test_interval_isequal - interval isequal: whether two boxes have the same bounds
//
// CORA: contSet/@interval/isequal.m
// CORA authors: Mark Wetzlinger
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
constexpr double kEps = std::numeric_limits<double>::epsilon();

void isequal_compares_the_bounds(const std::string &b) {
    const Interval I(vec({0.0, -1.0}), vec({1.0, 2.0}));
    check(I.isequal(I).all(), b + ": a box equals itself");
    check(!I.isequal(Interval(vec({0.0, -1.0}), vec({1.0, 2.5}))).all(), b + ": other bounds");
    check(I.isequal(Interval(vec({0.0, -1.0}), vec({1.0 + kEps, 2.0}))).all(),
          b + ": within CORA's default tol eps");
    check(!I.isequal(Interval(vec({0.0, -1.0}), vec({1.0 + 1e-9, 2.0}))).all(),
          b + ": 1e-9 apart is more than eps");
    check(I.isequal(Interval(vec({0.0, -1.0}), vec({1.0 + 1e-9, 2.0})), 1e-8).all(),
          b + ": but not for tol 1e-8");
    check(!I.isequal(Interval(vec({0.0}), vec({1.0}))).all(),
          b + ": another dimension is not equal");
    check(I.isequal(I, 0.0).all(), b + ": tol 0");
}

void isequal_with_infinite_bounds_points_and_batches(const std::string &b) {
    const Interval H(vec({0.0, -kInf}), vec({kInf, 1.0}));
    check(H.isequal(H).all(), b + ": the same infinities");
    // infinities against finite bounds
    check(!H.isequal(Interval(vec({0.0, -kInf}), vec({1e300, 1.0}))).all(),
          b + ": Inf against finite");
    check(!H.isequal(Interval(vec({0.0, -1e300}), vec({kInf, 1.0}))).all(),
          b + ": -Inf against finite");
    check(Interval::empty(2).isequal(Interval::empty(2)).all() &&
              !Interval::empty(2).isequal(H).all(),
          b + ": the empty interval");
    check(Interval(vec({1.0, 2.0}), vec({1.0, 2.0})).isequal(vec({1.0, 2.0})).all(),
          b + ": a point is [p, p]");
    const Interval I(vec({0.0}), vec({1.0})), J(vec({0.0}), vec({2.0}));
    check(Interval::stack({I, J}).isequal(Interval::stack({I, J})).all() &&
              !Interval::stack({I, J}).isequal(Interval::stack({I, I})).all(),
          b + ": a batch is equal only if every member is");
}

/// A negative tol and a point without an element axis are refused; the checks off give the same.
void checks_and_their_switch() {
    const Interval I(vec({0.0, -1.0}), vec({1.0, 2.0}));
    check(test::raises([&] { I.isequal(I, -1.0).all(); }, Error::wrong_value),
          "isequal: a negative tol");
    check(test::raises([&] { I.isequal(Tensor::from_data({1.0}, {1}).squeeze(0)).all(); },
                       Error::dimension_mismatch),
          "isequal: a point without an element axis");
    ChecksScope off(false);
    check(I.isequal(I).all(), "isequal: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        isequal_compares_the_bounds(b);
        isequal_with_infinite_bounds_points_and_batches(b);
    });
    checks_and_their_switch();
    return test::finish("interval isequal");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
