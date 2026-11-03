// test_empty_set_isequal - EmptySet isequal: equal to the empty sets of the same dimension
//
// CORA: contSet/@emptySet/isequal.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void equal_means_empty_in_the_same_dimension(const std::string &b) {
    const EmptySet E(2);
    check(E.isequal(EmptySet(2)), b + ": the same dimension");
    check(!E.isequal(EmptySet(3)), b + ": another dimension is false, not an error");
    check(E.isequal(Interval::empty(2)), b + ": the empty interval");
    check(!E.isequal(Interval::empty(3)), b + ": the empty interval of another dimension");
    check(!E.isequal(Interval(test::column({0.0, 0.0}), test::column({1.0, 1.0}))), b + ": a box");
    check(!E.isequal(Zonotope(test::column({0.0, 0.0}), Tensor::zeros({2, 0}))),
          b + ": a point zonotope");
    check(E.isequal(EmptySet(2), 1e-3), b + ": with a tol");
}

/// A negative tol is refused; with the checks off the answer is the same.
void checks_and_their_switch() {
    const EmptySet E(2);
    check(test::raises([&] { E.isequal(EmptySet(2), -1.0); }, Error::wrong_value),
          "isequal: tol = -1");
    ChecksScope off(false);
    check(E.isequal(EmptySet(2)), "isequal: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(equal_means_empty_in_the_same_dimension);
    checks_and_their_switch();
    return test::finish("empty_set isequal");
}
