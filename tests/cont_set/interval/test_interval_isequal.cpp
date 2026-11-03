// test_interval_isequal - interval isequal: whether two boxes have the same bounds
//
// CORA: contSet/@interval/isequal.m

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::column;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kEps = std::numeric_limits<double>::epsilon();

void isequal_compares_the_bounds(const std::string &b) {
    const Interval I(column({0.0, -1.0}), column({1.0, 2.0}));
    check(I.isequal(I), b + ": a box equals itself");
    check(!I.isequal(Interval(column({0.0, -1.0}), column({1.0, 2.5}))), b + ": other bounds");
    check(I.isequal(Interval(column({0.0, -1.0}), column({1.0 + kEps, 2.0}))),
          b + ": within CORA's default tol eps");
    check(!I.isequal(Interval(column({0.0, -1.0}), column({1.0 + 1e-9, 2.0}))),
          b + ": 1e-9 apart is more than eps");
    check(I.isequal(Interval(column({0.0, -1.0}), column({1.0 + 1e-9, 2.0})), 1e-8),
          b + ": but not for tol 1e-8");
    check(!I.isequal(Interval(column({0.0}), column({1.0}))), b + ": another dimension is not equal");
    check(I.isequal(I, 0.0), b + ": tol 0");
}

void isequal_with_infinite_bounds_points_and_batches(const std::string &b) {
    const Interval H(column({0.0, -kInf}), column({kInf, 1.0}));
    check(H.isequal(H), b + ": the same infinities");
    check(!H.isequal(Interval(column({0.0, -kInf}), column({1e300, 1.0}))), b + ": Inf against finite");
    check(!H.isequal(Interval(column({0.0, -1e300}), column({kInf, 1.0}))), b + ": -Inf against finite");
    check(Interval::empty(2).isequal(Interval::empty(2)) && !Interval::empty(2).isequal(H),
          b + ": the empty interval");
    check(Interval(column({1.0, 2.0}), column({1.0, 2.0})).isequal(column({1.0, 2.0})),
          b + ": a point is [p, p]");
    const Interval I(column({0.0}), column({1.0})), J(column({0.0}), column({2.0}));
    check(Interval::stack({I, J}).isequal(Interval::stack({I, J})) &&
              !Interval::stack({I, J}).isequal(Interval::stack({I, I})),
          b + ": a batch is equal only if every member is");
}

/// A negative tol and a point that is no matrix are refused; the checks off give the same.
void checks_and_their_switch() {
    const Interval I(column({0.0, -1.0}), column({1.0, 2.0}));
    check(test::raises([&] { I.isequal(I, -1.0); }, Error::wrong_value), "isequal: a negative tol");
    check(test::raises([&] { I.isequal(Tensor::from_data({1.0, 2.0}, {2})); },
                       Error::dimension_mismatch),
          "isequal: a point that is a flat array");
    ChecksScope off(false);
    check(I.isequal(I), "isequal: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        isequal_compares_the_bounds(b);
        isequal_with_infinite_bounds_points_and_batches(b);
    });
    checks_and_their_switch();
    return test::finish("interval isequal");
}
