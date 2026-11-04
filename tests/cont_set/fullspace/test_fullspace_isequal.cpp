// test_fullspace_isequal - fullspace isequal: whether a set is the whole space
//
// CORA: contSet/@fullspace/isequal.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::column;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

void isequal_recognizes_the_whole_space(const std::string &b) {
    const Fullspace F(2);
    check(F.isequal(Fullspace(2)) && !F.isequal(Fullspace(3)), b + ": another fullspace");
    check(F.isequal(Interval::unbounded(2)), b + ": the unbounded box");
    check(!F.isequal(Interval(column({-kInf, 0.0}), column({kInf, kInf}))),
          b + ": one finite bound");
    check(!F.isequal(Interval(column({-1e300, -1e300}), column({1e300, 1e300}))), b + ": huge");
    check(!F.isequal(Interval::empty(2)), b + ": the empty interval (CORA says true)");
    check(!F.isequal(Interval::unbounded(3)), b + ": another dimension");
    check(!F.isequal(Zonotope(column({0.0, 0.0}), Tensor::from_data({1, 0, 0, 1}, {2, 2}))),
          b + ": a zonotope");
    check(!F.isequal(Interval(column({0.0, 0.0}), column({0.0, 0.0}))), b + ": a point");
    check(F.isequal(Fullspace(2), 0.0) && F.isequal(Fullspace(2), 1.0), b + ": any tol");
}

/// A negative tol is refused; the checks off give the same answer.
void checks_and_their_switch() {
    const Fullspace F(2);
    check(test::raises([&] { F.isequal(F, -1.0); }, Error::wrong_value), "isequal: negative tol");
    ChecksScope off(false);
    check(F.isequal(F), "isequal: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(isequal_recognizes_the_whole_space);
    checks_and_their_switch();
    return test::finish("fullspace isequal");
}
